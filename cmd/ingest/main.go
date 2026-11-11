// Command ingest chunks the three papers at each chunk size, adds one chunk per table (from the
// arXiv LaTeX source), embeds every chunk with BGE, and writes to data/index/:
//
//	chunks_<size>.json   text + table chunks with paper / section / page
//	vectors_<size>.f32   float32 BGE vectors, one row per chunk, same order
//	tables.json          every table as structured rows and cells
//
//	go run -tags ORT ./cmd/ingest              full run (~4 min, mostly embedding)
//	go run -tags ORT ./cmd/ingest -tables-only only tables.json, no embedding
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
)

// Chunk sizes for retrieval settings A/C (300) and B (480). bge-base reads at most 512 tokens.
var sizes = []int{300, 480}

func main() {
	tablesOnly := flag.Bool("tables-only", false, "write data/index/tables.json and stop")
	flag.Parse()
	ctx := context.Background()
	start := time.Now()

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
		vf, err := os.Create(fmt.Sprintf("data/index/vectors_%d.f32", size))
		if err != nil {
			log.Fatal(err)
		}
		for i := 0; i < len(all); i += 32 {
			batch := all[i:min(i+32, len(all))]
			texts := make([]string, len(batch))
			for j, c := range batch {
				texts[j] = c.Text
			}
			vecs, err := e.Embed(ctx, texts)
			if err != nil {
				log.Fatalf("embed %s: %v", batch[0].ID, err)
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
		js, _ := json.MarshalIndent(all, "", "  ")
		if err := os.WriteFile(fmt.Sprintf("data/index/chunks_%d.json", size), js, 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("size %d: %d chunks written\n", size, len(all))
	}
	fmt.Printf("done in %s\n", time.Since(start).Round(time.Millisecond))
}
