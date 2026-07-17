import { useMemo, useState } from "react";
import type { Example } from "../types";
import type { ModalityTab } from "./Header";
import { modalityToTab } from "./Header";

export function ExampleBrowser({
  examples,
  modality,
  selectedId,
  onSelect,
}: {
  examples: Example[];
  modality: ModalityTab;
  selectedId: string | null;
  onSelect: (exampleId: string) => void;
}) {
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [format, setFormat] = useState("all");

  const forModality = useMemo(
    () => examples.filter((e) => modalityToTab(e.modality) === modality),
    [examples, modality],
  );
  const categories = useMemo(
    () => [...new Set(forModality.map((e) => e.category))].sort(),
    [forModality],
  );
  const formats = useMemo(
    () => [...new Set(forModality.map((e) => e.format))].sort(),
    [forModality],
  );

  const visible = forModality.filter((example) => {
    if (category !== "all" && example.category !== category) return false;
    if (format !== "all" && example.format !== format) return false;
    if (search) {
      const haystack =
        `${example.display_title} ${example.prompt_slug ?? ""} ${example.prompt_text ?? ""}`.toLowerCase();
      if (!haystack.includes(search.toLowerCase())) return false;
    }
    return true;
  });

  return (
    <section className="panel example-browser" aria-label="Example browser">
      <h2>Examples</h2>
      <div className="example-filters">
        <input
          type="search"
          placeholder="Search prompts…"
          aria-label="Search examples"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <select
          aria-label="Filter by category"
          value={category}
          onChange={(event) => setCategory(event.target.value)}
        >
          <option value="all">all categories</option>
          {categories.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
        <select
          aria-label="Filter by format"
          value={format}
          onChange={(event) => setFormat(event.target.value)}
        >
          <option value="all">all formats</option>
          {formats.map((value) => (
            <option key={value} value={value}>
              {value}
            </option>
          ))}
        </select>
      </div>
      {visible.length === 0 ? (
        <p className="muted">No examples match.</p>
      ) : (
        <ul className="example-list">
          {visible.map((example) => (
            <li key={example.example_id}>
              <button
                className={
                  example.example_id === selectedId
                    ? "example-item example-item-active"
                    : "example-item"
                }
                aria-pressed={example.example_id === selectedId}
                onClick={() => onSelect(example.example_id)}
              >
                <span className="example-title">{example.display_title}</span>
                <span className="example-meta">
                  <span className={`chip chip-${example.category}`}>
                    {example.category}
                  </span>
                  <span className="chip">{example.format}</span>
                  {example.strength && (
                    <span
                      className={`chip chip-strength-${example.strength.tag}`}
                      title={example.strength.basis}
                    >
                      {example.strength.tag}
                    </span>
                  )}
                  {example.data_status !== "measured" && (
                    <span className="chip chip-fixture">fixture</span>
                  )}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
