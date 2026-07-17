import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import App from "../App";
import { loadedData, testBundle } from "./fixture";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App with a preloaded text bundle", () => {
  it("renders header, provenance, and the default example", () => {
    render(<App preloaded={loadedData()} />);
    expect(
      screen.getByRole("heading", { name: /Gemma 4 Multimodal J-Lens Explorer/ }),
    ).toBeInTheDocument();
    expect(screen.getAllByText(/jspace_test/).length).toBeGreaterThan(0);
    expect(screen.getAllByText(/fa62d88df2e6/).length).toBeGreaterThan(0);
    // Text example auto-selected; its prompt is shown in the input viewer.
    expect(
      screen.getAllByText("The capital city of Australia is").length,
    ).toBeGreaterThan(0);
  });

  it("selects prompts from the example browser", async () => {
    const user = userEvent.setup();
    const data = loadedData();
    data.bundle.examples.push({
      ...data.bundle.examples[0],
      example_id: "text:antonym-early-late:1d78787985d0acdf",
      prompt_slug: "antonym-early-late",
      prompt_hash: "1d78787985d0acdf",
      category: "antonym",
      display_title: "The opposite of early is",
      prompt_text: "The opposite of early is",
    });
    render(<App preloaded={data} />);
    await user.click(screen.getByRole("button", { name: /The opposite of early is/ }));
    const viewer = screen.getByRole("region", { name: "Input viewer" });
    expect(within(viewer).getByText("The opposite of early is")).toBeInTheDocument();
  });

  it("filters examples by category", async () => {
    const user = userEvent.setup();
    render(<App preloaded={loadedData()} />);
    await user.selectOptions(screen.getByLabelText("Filter by category"), "factual");
    expect(
      screen.getByRole("button", { name: /The capital city of Australia is/ }),
    ).toBeInTheDocument();
  });

  it("switches token position and updates the prediction panel", async () => {
    const user = userEvent.setup();
    render(<App preloaded={loadedData()} />);
    // Default position is -1 with top-1 " Canberra".
    expect(screen.getByText("␣Canberra", { selector: ".token-big" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /-2: ␣Australia/ }));
    expect(screen.getByText("␣is", { selector: ".token-big" })).toBeInTheDocument();
  });

  it("switches layers via the layer rail", async () => {
    const user = userEvent.setup();
    render(<App preloaded={loadedData()} />);
    // Default layer is the highest (38).
    expect(screen.getByText(/Sparse cone \(k=10, layer 38/)).toBeInTheDocument();
    const rail = screen.getByRole("navigation", { name: "Layer selection" });
    await user.click(within(rail).getByRole("button", { name: /L35/ }));
    expect(screen.getByText(/Sparse cone \(k=10, layer 35/)).toBeInTheDocument();
  });

  it("renders the cone with output-token and frequency markers", () => {
    render(<App preloaded={loadedData()} />);
    const conePanel = screen.getByRole("region", { name: "Sparse cone" });
    expect(within(conePanel).getByText("★ output")).toBeInTheDocument();
    expect(within(conePanel).getByText("◆ frequent")).toBeInTheDocument();
    expect(within(conePanel).getByText("19.00%")).toBeInTheDocument(); // explained
  });

  it("advances pursuit playback steps", async () => {
    const user = userEvent.setup();
    render(<App preloaded={loadedData()} />);
    const panel = screen.getByRole("region", { name: "Gradient pursuit playback" });
    expect(within(panel).getByText(/step 0 \/ 2/)).toBeInTheDocument();
    expect(within(panel).getByText(/per-step coefficients were not recorded/)).toBeInTheDocument();
    await user.click(within(panel).getByRole("button", { name: "Next step" }));
    expect(within(panel).getByText(/step 1 \/ 2/)).toBeInTheDocument();
    expect(within(panel).getByText("␣Canberra", { selector: ".pursuit-added" })).toBeInTheDocument();
    await user.click(within(panel).getByRole("button", { name: "Next step" }));
    expect(within(panel).getByText(/step 2 \/ 2/)).toBeInTheDocument();
    expect(within(panel).getByRole("button", { name: "Next step" })).toBeDisabled();
  });

  it("renders the cross-layer trajectory with transition stats", () => {
    render(<App preloaded={loadedData()} />);
    const panel = screen.getByRole("region", { name: "Cross-layer trajectory" });
    expect(within(panel).getByText("L35")).toBeInTheDocument();
    expect(within(panel).getByText("L38")).toBeInTheDocument();
    expect(within(panel).getByText("0.33")).toBeInTheDocument(); // jaccard
    expect(within(panel).getByText(/\+1 entered/)).toBeInTheDocument();
    expect(within(panel).getByText(/1 retained/)).toBeInTheDocument();
  });

  it("shows measured causal records and switches multipliers", async () => {
    const user = userEvent.setup();
    render(<App preloaded={loadedData()} />);
    const panel = screen.getByRole("region", { name: "Causal steering" });
    // Only measured multipliers are offered.
    const group = within(panel).getByRole("group", { name: "Measured multipliers" });
    expect(within(group).getByRole("button", { name: "-1" })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: "0" })).toBeInTheDocument();
    expect(within(group).getByRole("button", { name: "+1" })).toBeInTheDocument();
    expect(within(group).queryByRole("button", { name: "+0.5" })).toBeNull();

    await user.click(within(group).getByRole("button", { name: "-1" }));
    // Targeted readout shows the flip and the matched control appears.
    expect(within(panel).getAllByText(/Measured/).length).toBeGreaterThan(0);
    expect(within(panel).getByText(/Matched control/)).toBeInTheDocument();
    const readouts = within(panel).getAllByText(/target logit Δ/);
    expect(readouts.length).toBe(2); // targeted + control
  });

  it("shows the no-causal-data state for examples without records", async () => {
    const user = userEvent.setup();
    const data = loadedData();
    data.bundle.causal_records = [];
    render(<App preloaded={data} />);
    expect(screen.getByText("No causal data available")).toBeInTheDocument();
    expect(screen.getByText(/No intervention records exist/)).toBeInTheDocument();
    expect(user).toBeTruthy();
  });

  it("badges fixture-status examples distinctly", async () => {
    const user = userEvent.setup();
    render(<App preloaded={loadedData()} />);
    // Text example is measured.
    expect(screen.getByText(/Measured · current example/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Image" }));
    expect(screen.getByText(/Synthetic UI fixture · current example/)).toBeInTheDocument();
  });
});

describe("modality switching", () => {
  it("renders the image example with its asset and metadata", async () => {
    const user = userEvent.setup();
    render(<App preloaded={loadedData()} />);
    await user.click(screen.getByRole("button", { name: "Image" }));
    const viewer = screen.getByRole("region", { name: "Input viewer" });
    const image = within(viewer).getByRole("img", { name: "The dominant color is" });
    expect(image).toHaveAttribute("src", "data/fixtures/assets/fixture_image.png");
    expect(within(viewer).getByText(/96×64px/)).toBeInTheDocument();
    expect(within(viewer).getByText(/image tokens at sequence positions \[1, 257\)/)).toBeInTheDocument();
  });

  it("renders the audio example with a native player", async () => {
    const user = userEvent.setup();
    const { container } = render(<App preloaded={loadedData()} />);
    await user.click(screen.getByRole("button", { name: "Audio" }));
    const audio = container.querySelector("audio");
    expect(audio).not.toBeNull();
    expect(audio).toHaveAttribute("src", "data/fixtures/assets/fixture_audio.wav");
    expect(screen.getByText(/0\.4 s/)).toBeInTheDocument();
    expect(screen.getByText(/16000 Hz/)).toBeInTheDocument();
  });

  it("shows an empty state when a modality has no examples", async () => {
    const user = userEvent.setup();
    const data = loadedData();
    data.bundle.examples = data.bundle.examples.filter((e) => e.modality === "text");
    render(<App preloaded={data} />);
    await user.click(screen.getByRole("button", { name: /Image/ }));
    expect(screen.getByText(/No image-conditioned records yet/)).toBeInTheDocument();
  });
});

describe("bundle loading over fetch", () => {
  it("loads the text bundle and prefers fixtures only when measured is absent", async () => {
    const text = testBundle();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (String(url).endsWith("text_demo.json")) {
          return { ok: true, json: async () => text } as Response;
        }
        return { ok: false, json: async () => ({}) } as Response;
      }),
    );
    render(<App preloaded={undefined} />);
    await waitFor(() =>
      expect(
        screen.getAllByText("The capital city of Australia is").length,
      ).toBeGreaterThan(0),
    );
  });

  it("shows a clear error state on a malformed bundle", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({
        ok: true,
        json: async () => ({ schema: "wrong.thing" }),
      })) as unknown as typeof fetch,
    );
    render(<App preloaded={undefined} />);
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        /unsupported bundle schema/,
      ),
    );
  });

  it("shows a clear error when the demo bundle is missing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, json: async () => ({}) })) as unknown as typeof fetch,
    );
    render(<App preloaded={undefined} />);
    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent(
        /could not load data\/text_demo\.json/,
      ),
    );
  });
});
