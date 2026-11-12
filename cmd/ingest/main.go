// Command ingest chunks the three papers at each chunk size, adds one chunk per table (from the
// arXiv LaTeX source), embeds every chunk with BGE, and writes to data/index/:
//
//	chunks_<size>.json   text + table chunks with paper / section / page
//	vectors_<size>.f32   float32 BGE vectors, one row per chunk, same order
//	tables.json          every table as structured rows and cells
//	windows_300.json/.f32 2-sentence windows for setting C (small-to-big), parent = chunk index
//
//	go run -tags ORT ./cmd/ingest              full run (~4 min, mostly embedding)
//	go run -tags ORT ./cmd/ingest -tables-only only tables.json, no embedding
//	go run -tags ORT ./cmd/ingest -windows-only rebuild windows_300.* from chunks_300.json
package main

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/Praroop1435/Vaani/internal/embed"
	"github.com/Praroop1435/Vaani/internal/ingest"
	"github.com/Praroop1435/Vaani/internal/retrieve"
)

// Chunk sizes for retrieval settings A/C (300) and B (480). bge-base reads at most 512 tokens.
var sizes = []int{300, 480}

func main() {
	tablesOnly := flag.Bool("tables-only", false, "write data/index/tables.json and stop")
	windowsOnly := flag.Bool("windows-only", false, "rebuild windows from the existing chunks_300.json and stop")
	metaOnly := flag.Bool("meta-only", false, "rebuild vectors_meta_300.f32 (setting R) from the existing index and stop")
	flag.Parse()
	ctx := context.Background()
	start := time.Now()
	if *metaOnly {
		e, err := embed.New(ctx, embed.DefaultModelDir)
		if err != nil {
			log.Fatal(err)
		}
		defer e.Close()
		if err := retrieve.BuildMeta(ctx, e, "data/index", windowParent); err != nil {
			log.Fatal(err)
		}
		fmt.Println("wrote data/index/vectors_meta_300.f32 in", time.Since(start).Round(time.Second))
		return
	}
	if *windowsOnly {
		e, err := embed.New(ctx, embed.DefaultModelDir)
		if err != nil {
			log.Fatal(err)
		}
		defer e.Close()
		js, err := os.ReadFile(fmt.Sprintf("data/index/chunks_%d.json", windowParent))
		if err != nil {
			log.Fatal(err)
		}
		var chunks []ingest.Chunk
		if err := json.Unmarshal(js, &chunks); err != nil {
			log.Fatal(err)
		}
		buildWindows(ctx, e, chunks)
		fmt.Printf("done in %s\n", time.Since(start).Round(time.Millisecond))
		return
	}

	type parsed struct {
		paper  ingest.Paper
		segs   []ingest.Segment
		tables []ingest.Table
	}
	var docs []parsed
	var allTables []ingest.Table
	for _, p := range ingest.Papers {
		pages, outline, err := ingest.Extract(filepath.Join("data/papers", p.ID+".pdf"))
		if err != nil {
			log.Fatal(err)
		}
		heads, err := ingest.ParseOutline(outline)
		if err != nil {
			log.Fatal(p.ID, ": ", err)
		}
		segs, warns := ingest.Segments(pages, heads, p.Extra)

		srcDir := filepath.Join("data/latex", p.ID)
		if err := ingest.FetchSource(p.ID, srcDir); err != nil {
			log.Fatal(err)
		}
		src, err := ingest.LoadSource(srcDir)
		if err != nil {
			log.Fatal(err)
		}
		tables, tw := ingest.ParseTables(p.ID, src)
		warns = append(warns, tw...)
		warns = append(warns, ingest.LocateTables(tables, pages, segs)...)

		fmt.Printf("%s: %d pages, %d outline headings, %d sections, %d tables, %d warnings\n", p.ID, len(pages), len(heads), len(segs), len(tables), len(warns))
		for _, w := range warns {
			fmt.Println("  warning:", w)
		}
		var located []ingest.Table // a table in the source but not in the PDF cannot be cited
		for _, t := range tables {
			if t.Number != "" {
				located = append(located, t)
			}
		}
		docs = append(docs, parsed{p, segs, located})
		allTables = append(allTables, located...)
	}

	if err := os.MkdirAll("data/index", 0o755); err != nil {
		log.Fatal(err)
	}
	js, _ := json.MarshalIndent(allTables, "", "  ")
	if err := os.WriteFile("data/index/tables.json", js, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("tables.json: %d tables\n", len(allTables))
	if *tablesOnly {
		return
	}

	e, err := embed.New(ctx, embed.DefaultModelDir)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()

	for _, size := range sizes {
		var all []ingest.Chunk
		for _, d := range docs {
			cs := ingest.Pack(d.paper.ID, d.segs, size, embed.MaxTokens, e.CountTokens)
			ts := ingest.TableChunks(d.tables, embed.MaxTokens, e.CountTokens, len(cs))
			fmt.Printf("size %d: %s -> %d text + %d table chunks\n", size, d.paper.ID, len(cs), len(ts))
			all = append(all, cs...)
			all = append(all, ts...)
		}
		texts := make([]string, len(all))
		for i, c := range all {
			texts[i] = c.Text
		}
		writeVectors(ctx, e, texts, fmt.Sprintf("data/index/vectors_%d.f32", size))
		js, _ := json.MarshalIndent(all, "", "  ")
		if err := os.WriteFile(fmt.Sprintf("data/index/chunks_%d.json", size), js, 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("size %d: %d chunks written\n", size, len(all))
		if size == windowParent {
			buildWindows(ctx, e, all)
		}
	}
	fmt.Printf("done in %s\n", time.Since(start).Round(time.Millisecond))
}

// Setting C searches 2-sentence windows (stride 1) cut from the 300-token chunks.
const windowParent, windowSize, windowStride = 300, 2, 1

func buildWindows(ctx context.Context, e *embed.Embedder, chunks []ingest.Chunk) {
	ws := ingest.Windows(chunks, windowSize, windowStride)
	texts := make([]string, len(ws))
	for i, w := range ws {
		texts[i] = w.Text
	}
	writeVectors(ctx, e, texts, fmt.Sprintf("data/index/windows_%d.f32", windowParent))
	js, _ := json.MarshalIndent(ws, "", "  ")
	if err := os.WriteFile(fmt.Sprintf("data/index/windows_%d.json", windowParent), js, 0o644); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("windows: %d from %d chunks\n", len(ws), len(chunks))
}

// writeVectors embeds texts in batches and writes float32 rows, one per text.
func writeVectors(ctx context.Context, e *embed.Embedder, texts []string, path string) {
	vf, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	for i := 0; i < len(texts); i += 32 {
		vecs, err := e.Embed(ctx, texts[i:min(i+32, len(texts))])
		if err != nil {
			log.Fatalf("embed row %d of %s: %v", i, path, err)
		}
		for _, v := range vecs {
			if err := binary.Write(vf, binary.LittleEndian, v); err != nil {
				log.Fatal(err)
			}
		}
	}
	if err := vf.Close(); err != nil {
		log.Fatal(err)
	}
}
