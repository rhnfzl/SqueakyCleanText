#!/usr/bin/env python3
"""Comprehensive A/B test: Torch vs ONNX NER pipelines.

Uses a 200+ sentence hand-crafted corpus covering all entity types, languages,
edge cases, and text lengths. Measures entity accuracy AND inference speed.

Requires: torch, transformers, onnxruntime (all in current env)
"""

import json
import time
import statistics

import numpy as np

# ── Torch (HuggingFace) pipeline ──
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline as hf_pipeline
import torch

# ── ONNX pipeline (our implementation) ──
from sct.utils.onnx_pipeline import load_onnx_ner_model

# ═══════════════════════════════════════════════════════════════════════════════
# Comprehensive test corpus: 200+ sentences across categories
# ═══════════════════════════════════════════════════════════════════════════════

# Category 1: English — persons (40 sentences)
PERSON_TEXTS = [
    "John Smith works at the downtown office every weekday.",
    "Dr. Sarah Wilson published groundbreaking research in neuroscience.",
    "Barack Obama served as the 44th President of the United States.",
    "Elon Musk announced new plans for SpaceX yesterday.",
    "Queen Elizabeth II reigned for over 70 years.",
    "Albert Einstein developed the theory of relativity.",
    "Marie Curie won two Nobel Prizes in different sciences.",
    "Leonardo da Vinci painted the Mona Lisa in Florence.",
    "William Shakespeare wrote Hamlet around 1600.",
    "Mahatma Gandhi led India's independence movement.",
    "Nelson Mandela was imprisoned for 27 years.",
    "Martin Luther King Jr. delivered his famous speech in Washington.",
    "Oprah Winfrey became one of the most influential media personalities.",
    "Steve Jobs co-founded Apple Computer in his garage.",
    "Mark Zuckerberg created Facebook while at Harvard University.",
    "Jeff Bezos started Amazon as an online bookstore.",
    "Bill Gates and Paul Allen founded Microsoft together.",
    "Warren Buffett is often called the Oracle of Omaha.",
    "Taylor Swift broke multiple music industry records.",
    "Cristiano Ronaldo scored his 800th career goal.",
    "Lionel Messi moved from Barcelona to Paris Saint-Germain.",
    "Serena Williams dominated women's tennis for decades.",
    "Michael Jordan is considered the greatest basketball player.",
    "Tiger Woods won his first major championship at age 21.",
    "Usain Bolt holds the world record in the 100 meters.",
    "Stephen Hawking published A Brief History of Time.",
    "Jane Austen wrote Pride and Prejudice in the early 1800s.",
    "Charles Dickens described Victorian London in vivid detail.",
    "Pablo Picasso revolutionized modern art with cubism.",
    "Frida Kahlo became an icon of Mexican culture.",
    "Wolfgang Amadeus Mozart composed his first symphony at age eight.",
    "Ludwig van Beethoven continued composing after losing his hearing.",
    "Nikola Tesla invented the alternating current electrical system.",
    "Alexander Fleming discovered penicillin by accident.",
    "Ada Lovelace is often considered the first computer programmer.",
    "Alan Turing cracked the Enigma code during World War II.",
    "Malala Yousafzai became the youngest Nobel Peace Prize laureate.",
    "Greta Thunberg led a global climate change movement.",
    "Pope Francis became the first pope from the Americas.",
    "Satya Nadella transformed Microsoft's cloud business strategy.",
]

# Category 2: English — organizations (30 sentences)
ORG_TEXTS = [
    "Microsoft Corporation announced quarterly earnings exceeding expectations.",
    "Google LLC developed a new artificial intelligence model.",
    "Apple Inc. released the latest iPhone at their keynote event.",
    "Amazon Web Services dominates the cloud computing market.",
    "The United Nations held an emergency session on climate change.",
    "NATO expanded its presence in Eastern Europe.",
    "The European Union imposed new sanctions on trade violations.",
    "FIFA announced the next World Cup host nation.",
    "The World Health Organization declared a global health emergency.",
    "NASA launched the Artemis mission to the Moon.",
    "SpaceX successfully landed its Starship rocket.",
    "The Federal Reserve raised interest rates again.",
    "Goldman Sachs reported strong investment banking revenues.",
    "JPMorgan Chase acquired a major fintech startup.",
    "Toyota Motor Corporation unveiled its electric vehicle lineup.",
    "Samsung Electronics launched a new foldable smartphone.",
    "The Red Cross provided humanitarian aid to the region.",
    "Amnesty International published a report on human rights.",
    "The International Olympic Committee selected the host city.",
    "Harvard University announced a new scholarship program.",
    "Stanford University researchers made a breakthrough in AI.",
    "Oxford University Press published the updated dictionary.",
    "Tesla Motors built a new Gigafactory in Texas.",
    "The BBC reported on the developing situation.",
    "Reuters and Associated Press covered the breaking news.",
    "The New York Times won multiple Pulitzer Prizes.",
    "Pfizer and BioNTech developed the mRNA vaccine.",
    "The World Bank provided funding for infrastructure projects.",
    "Greenpeace activists protested against oil drilling.",
    "The International Monetary Fund released its economic forecast.",
]

# Category 3: English — locations (30 sentences)
LOCATION_TEXTS = [
    "The conference was held in New York City last week.",
    "Silicon Valley remains the world's leading technology hub.",
    "London's financial district is known as the City of London.",
    "The Eiffel Tower in Paris attracts millions of visitors.",
    "Mount Everest is the highest peak in the Himalayas.",
    "The Amazon River flows through Brazil and Peru.",
    "Tokyo hosted the Olympic Games in 2021.",
    "Sydney's Opera House is an architectural masterpiece.",
    "The Great Wall of China stretches thousands of kilometers.",
    "Moscow's Red Square is a famous landmark.",
    "The Sahara Desert spans across Northern Africa.",
    "The Mediterranean Sea borders three continents.",
    "Wall Street is located in Lower Manhattan.",
    "The Pentagon is headquartered in Arlington, Virginia.",
    "Cape Town sits at the southern tip of Africa.",
    "Dubai has become a global business and tourism center.",
    "Singapore is both a city and a country.",
    "The Nile River flows through Egypt to the Mediterranean.",
    "The Rocky Mountains extend from Canada to New Mexico.",
    "Antarctica is the coldest continent on Earth.",
    "The Pacific Ocean is the largest body of water.",
    "Istanbul straddles both Europe and Asia.",
    "Jerusalem is a holy city for three religions.",
    "The Grand Canyon in Arizona is a natural wonder.",
    "Venice is famous for its canals and architecture.",
    "Machu Picchu sits high in the Andes Mountains of Peru.",
    "The Danube River flows through ten European countries.",
    "San Francisco is known for the Golden Gate Bridge.",
    "The North Pole is located in the Arctic Ocean.",
    "Kyoto was Japan's capital for over a thousand years.",
]

# Category 4: Multi-entity texts (30 sentences)
MULTI_ENTITY_TEXTS = [
    "John Smith from Microsoft visited the Google campus in Mountain View.",
    "Angela Merkel met Emmanuel Macron at the Elysee Palace in Paris.",
    "Tim Cook announced that Apple would invest in a new factory in Austin, Texas.",
    "Warren Buffett's Berkshire Hathaway bought shares of Bank of America.",
    "The World Economic Forum in Davos hosted leaders from around the globe.",
    "Sundar Pichai led Google's expansion into artificial intelligence research.",
    "Elon Musk's Tesla and SpaceX both have facilities in Hawthorne, California.",
    "Jeff Bezos stepped down as CEO of Amazon and moved to Blue Origin.",
    "The Nobel Prize committee in Stockholm honored researchers from MIT.",
    "President Biden addressed Congress in Washington about NATO commitments.",
    "Boris Johnson left 10 Downing Street after his tenure as Prime Minister.",
    "Xi Jinping visited the United Nations headquarters in New York.",
    "The G7 summit in Hiroshima brought together leaders from seven nations.",
    "Mark Zuckerberg testified before the European Parliament in Brussels.",
    "Sergey Brin and Larry Page founded Google in a garage in Menlo Park.",
    "Jack Ma's Alibaba Group is headquartered in Hangzhou, China.",
    "Rishi Sunak became Prime Minister of the United Kingdom.",
    "Volodymyr Zelensky addressed the European Council in Strasbourg.",
    "Christine Lagarde leads the European Central Bank in Frankfurt.",
    "Janet Yellen met with officials at the International Monetary Fund.",
    "Sam Altman's OpenAI released ChatGPT from their office in San Francisco.",
    "Jensen Huang presented NVIDIA's new chip at the Consumer Electronics Show in Las Vegas.",
    "Satya Nadella and Sam Altman discussed AI strategy at Microsoft's headquarters in Redmond.",
    "The Red Cross deployed teams to Turkey and Syria after the earthquake.",
    "UNICEF provided supplies to children in the Democratic Republic of Congo.",
    "The Olympic torch was carried through the streets of Tokyo and Osaka.",
    "Astronauts from NASA and ESA conducted experiments on the International Space Station.",
    "The Vatican's Pope Francis traveled from Rome to Lisbon for World Youth Day.",
    "Roger Federer retired from tennis after playing his final match at the Laver Cup in London.",
    "Lewis Hamilton moved from Mercedes to Ferrari for the next Formula One season.",
]

# Category 5: German texts (15 sentences)
GERMAN_TEXTS = [
    "Angela Merkel besuchte die Technische Universität Berlin.",
    "Olaf Scholz traf sich mit Emmanuel Macron in Paris.",
    "Die Deutsche Bank hat ihren Hauptsitz in Frankfurt am Main.",
    "BMW produziert Autos in München und Leipzig.",
    "Sebastian Vettel gewann vier Weltmeisterschaften in der Formel Eins.",
    "Thomas Müller spielt für den FC Bayern München.",
    "Albert Einstein wurde in Ulm geboren und lebte in Zürich.",
    "Siemens und Bosch sind große deutsche Industrieunternehmen.",
    "Der Bundestag in Berlin debattierte über neue Gesetze.",
    "Goethe schrieb Faust in Weimar.",
    "Beethoven wurde in Bonn geboren und lebte in Wien.",
    "Die Berliner Mauer fiel im November in der deutschen Hauptstadt.",
    "Volkswagen hat Werke in Wolfsburg und Dresden.",
    "Martin Luther begann die Reformation in Wittenberg.",
    "Karl Marx wurde in Trier geboren und lebte in London.",
]

# Category 6: Dutch texts (15 sentences)
DUTCH_TEXTS = [
    "Willem-Alexander woont in Den Haag en bezoekt Amsterdam regelmatig.",
    "Ajax speelde tegen Feyenoord in Rotterdam.",
    "Rembrandt van Rijn schilderde de Nachtwacht in Amsterdam.",
    "Philips heeft zijn hoofdkantoor in Eindhoven.",
    "Max Verstappen won de Formule Eén Grand Prix in Zandvoort.",
    "De Rijksmuseum staat in het centrum van Amsterdam.",
    "Vincent van Gogh werd geboren in Zundert en woonde in Arles.",
    "Mark Rutte was premier van Nederland.",
    "De Erasmus Universiteit staat in Rotterdam.",
    "Shell heeft kantoren in Den Haag en Londen.",
    "Johan Cruyff speelde voor Ajax en FC Barcelona.",
    "Het Mauritshuis in Den Haag toont werken van Vermeer.",
    "De TU Delft is een bekende technische universiteit.",
    "Anne Frank schreef haar dagboek in Amsterdam.",
    "De Tweede Kamer vergaderde in het Binnenhof in Den Haag.",
]

# Category 7: Spanish texts (15 sentences)
SPANISH_TEXTS = [
    "Pablo García trabaja en Madrid para Google España.",
    "Lionel Messi jugó muchos años en el FC Barcelona.",
    "El Real Madrid ganó la Liga de Campeones en París.",
    "Gabriel García Márquez escribió Cien Años de Soledad en Colombia.",
    "Pedro Sánchez es el presidente del gobierno de España.",
    "Telefónica tiene su sede central en Madrid.",
    "Rafael Nadal ganó Roland Garros en París múltiples veces.",
    "El Museo del Prado está en el centro de Madrid.",
    "Carlos Slim es uno de los hombres más ricos de México.",
    "La Universidad de Buenos Aires es la más grande de Argentina.",
    "Fernando Alonso compitió en Fórmula Uno para Ferrari y McLaren.",
    "Frida Kahlo y Diego Rivera vivieron en Ciudad de México.",
    "El Banco Santander tiene oficinas en toda América Latina.",
    "Pablo Neruda recibió el Premio Nobel de Literatura.",
    "La Sagrada Familia de Gaudí está en Barcelona.",
]

# Category 8: Edge cases (20 sentences)
EDGE_CASES = [
    # No entities
    "The quick brown fox jumps over the lazy dog.",
    "It was a dark and stormy night with heavy rain.",
    "Mathematics and physics are fundamental sciences.",
    # Very short
    "John.",
    "Paris.",
    "Google.",
    # Adjacent entities
    "John Smith Sarah Johnson met in the park.",
    "Microsoft Google Apple are technology companies.",
    "London Paris Berlin are European capitals.",
    # Subword challenges
    "Arnold Schwarzenegger visited Liechtenstein.",
    "Tchaikovsky composed the Nutcracker ballet.",
    "Dostoevsky wrote Crime and Punishment.",
    # Unicode / accented
    "José María González lives in São Paulo.",
    "François Mitterrand led France from the Élysée Palace.",
    "Müller and Günther work at Böhringer Ingelheim in Düsseldorf.",
    # Possessives and punctuation
    "Microsoft's CEO spoke at Harvard.",
    "Obama's presidency changed America's political landscape.",
    # Abbreviations
    "U.N. Secretary-General met with E.U. officials.",
    "The U.S.A. and U.K. signed a trade agreement.",
    # Hyphenated names
    "Jean-Claude Van Damme visited Saint-Germain-des-Prés in Paris.",
]

# Combine all categories
ALL_TEXTS = (
    PERSON_TEXTS + ORG_TEXTS + LOCATION_TEXTS + MULTI_ENTITY_TEXTS
    + GERMAN_TEXTS + DUTCH_TEXTS + SPANISH_TEXTS + EDGE_CASES
)

# ═══════════════════════════════════════════════════════════════════════════════


def format_entities(entities):
    """Format entity list for display."""
    parts = []
    for e in entities:
        parts.append(f"({e['entity_group']}, '{e['word']}', {e['score']:.4f})")
    return "[" + ", ".join(parts) + "]"


def run_pipeline(pipe, text, is_torch=False):
    """Run a pipeline and return normalized entity dicts."""
    if is_torch:
        with torch.no_grad():
            results = pipe(text)
    else:
        results = pipe([text])[0]

    return [
        {
            "entity_group": r["entity_group"],
            "word": r["word"],
            "start": r["start"],
            "end": r["end"],
            "score": float(r["score"]),
        }
        for r in results
    ]


def compare_entities(torch_entities, onnx_entities):
    """Compare two entity lists and return detailed metrics."""
    torch_set = {(e["entity_group"], e["start"], e["end"]) for e in torch_entities}
    onnx_set = {(e["entity_group"], e["start"], e["end"]) for e in onnx_entities}

    labels_match = torch_set == onnx_set

    # Score comparison (for matching entities)
    score_diffs = []
    torch_by_span = {(e["start"], e["end"]): e for e in torch_entities}
    onnx_by_span = {(e["start"], e["end"]): e for e in onnx_entities}
    for span in torch_by_span:
        if span in onnx_by_span:
            diff = abs(torch_by_span[span]["score"] - onnx_by_span[span]["score"])
            score_diffs.append(diff)

    return {
        "labels_match": labels_match,
        "torch_only": torch_set - onnx_set,
        "onnx_only": onnx_set - torch_set,
        "score_diffs": score_diffs,
        "torch_count": len(torch_entities),
        "onnx_count": len(onnx_entities),
    }


def benchmark_speed(pipe, texts, is_torch=False, warmup=3, runs=5):
    """Benchmark inference speed with warmup and multiple timed runs."""
    # Warmup
    for _ in range(warmup):
        for text in texts[:10]:
            run_pipeline(pipe, text, is_torch)

    # Timed runs
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        for text in texts:
            run_pipeline(pipe, text, is_torch)
        elapsed = time.perf_counter() - start
        timings.append(elapsed)

    return {
        "total_texts": len(texts),
        "runs": runs,
        "mean_total_s": statistics.mean(timings),
        "std_total_s": statistics.stdev(timings) if len(timings) > 1 else 0,
        "min_total_s": min(timings),
        "max_total_s": max(timings),
        "mean_per_text_ms": 1000 * statistics.mean(timings) / len(texts),
        "p50_per_text_ms": 1000 * statistics.median(timings) / len(texts),
        "texts_per_second": len(texts) / statistics.mean(timings),
        "raw_timings": timings,
    }


def main():
    torch_model_name = "dslim/bert-base-NER"
    onnx_model_name = "protectai/bert-base-NER-onnx"

    print("=" * 90)
    print("COMPREHENSIVE A/B TEST: HuggingFace Torch vs ONNX NER Pipeline")
    print("=" * 90)
    print(f"\nCorpus: {len(ALL_TEXTS)} texts across 8 categories")
    print(f"  Persons:      {len(PERSON_TEXTS)}")
    print(f"  Organizations:{len(ORG_TEXTS)}")
    print(f"  Locations:    {len(LOCATION_TEXTS)}")
    print(f"  Multi-entity: {len(MULTI_ENTITY_TEXTS)}")
    print(f"  German:       {len(GERMAN_TEXTS)}")
    print(f"  Dutch:        {len(DUTCH_TEXTS)}")
    print(f"  Spanish:      {len(SPANISH_TEXTS)}")
    print(f"  Edge cases:   {len(EDGE_CASES)}")

    # ── Load models ──
    print("\n[1/4] Loading models...")
    print(f"  Torch: {torch_model_name}")
    tok = AutoTokenizer.from_pretrained(torch_model_name)
    model = AutoModelForTokenClassification.from_pretrained(torch_model_name)
    torch_pipe = hf_pipeline(
        "ner", model=model, tokenizer=tok,
        aggregation_strategy="simple", device="cpu"
    )

    print(f"  ONNX:  {onnx_model_name}")
    onnx_pipe, _ = load_onnx_ner_model(onnx_model_name, device="cpu")

    # ── Entity Accuracy Comparison ──
    print("\n[2/4] Running entity accuracy comparison...")
    exact_matches = 0
    label_matches = 0
    total = 0
    all_score_diffs = []
    mismatches = []
    entity_count_torch = 0
    entity_count_onnx = 0
    category_stats = {}

    # Track per-category stats
    categories = [
        ("Persons", PERSON_TEXTS),
        ("Organizations", ORG_TEXTS),
        ("Locations", LOCATION_TEXTS),
        ("Multi-entity", MULTI_ENTITY_TEXTS),
        ("German", GERMAN_TEXTS),
        ("Dutch", DUTCH_TEXTS),
        ("Spanish", SPANISH_TEXTS),
        ("Edge cases", EDGE_CASES),
    ]

    text_idx = 0
    for cat_name, cat_texts in categories:
        cat_exact = 0
        cat_label = 0
        cat_total = 0
        for text in cat_texts:
            torch_ents = run_pipeline(torch_pipe, text, is_torch=True)
            onnx_ents = run_pipeline(onnx_pipe, text, is_torch=False)

            comp = compare_entities(torch_ents, onnx_ents)
            total += 1
            cat_total += 1
            entity_count_torch += comp["torch_count"]
            entity_count_onnx += comp["onnx_count"]

            if comp["labels_match"]:
                label_matches += 1
                cat_label += 1

            all_score_diffs.extend(comp["score_diffs"])
            scores_close = all(d < 0.01 for d in comp["score_diffs"]) if comp["score_diffs"] else True

            if comp["labels_match"] and scores_close:
                exact_matches += 1
                cat_exact += 1
            else:
                mismatches.append({
                    "index": text_idx,
                    "category": cat_name,
                    "text": text[:100],
                    "torch_ents": format_entities(torch_ents),
                    "onnx_ents": format_entities(onnx_ents),
                    "torch_only": [str(x) for x in comp["torch_only"]],
                    "onnx_only": [str(x) for x in comp["onnx_only"]],
                    "max_score_diff": max(comp["score_diffs"]) if comp["score_diffs"] else None,
                })

            text_idx += 1

        category_stats[cat_name] = {
            "total": cat_total,
            "exact": cat_exact,
            "label": cat_label,
            "pct": f"{100 * cat_exact / cat_total:.1f}%" if cat_total else "N/A",
        }
        print(f"  {cat_name}: {cat_exact}/{cat_total} exact ({category_stats[cat_name]['pct']})")

    # ── Speed Benchmark ──
    print("\n[3/4] Running speed benchmarks...")

    # Use first 50 texts (mix of categories) for speed
    speed_texts = ALL_TEXTS[:50]
    print(f"  Benchmarking on {len(speed_texts)} texts (3 warmup + 5 timed runs)...")

    torch_speed = benchmark_speed(torch_pipe, speed_texts, is_torch=True, warmup=3, runs=5)
    print(f"  Torch done: {torch_speed['mean_per_text_ms']:.2f} ms/text")

    onnx_speed = benchmark_speed(onnx_pipe, speed_texts, is_torch=False, warmup=3, runs=5)
    print(f"  ONNX  done: {onnx_speed['mean_per_text_ms']:.2f} ms/text")

    onnx_ms = onnx_speed["mean_per_text_ms"]
    speedup = torch_speed["mean_per_text_ms"] / onnx_ms if onnx_ms > 0 else float("inf")

    # ── Report ──
    print(f"\n{'=' * 90}")
    print("[4/4] RESULTS")
    print("=" * 90)

    print(f"\n{'─' * 50}")
    print("ENTITY ACCURACY")
    print(f"{'─' * 50}")
    print(f"  Total texts tested:   {total}")
    print(f"  Torch entities found: {entity_count_torch}")
    print(f"  ONNX entities found:  {entity_count_onnx}")
    print(f"  Entity count match:   {'YES' if entity_count_torch == entity_count_onnx else 'NO'}")
    print(f"  Exact match:          {exact_matches}/{total} ({100 * exact_matches / total:.1f}%)")
    print(f"  Label+span match:     {label_matches}/{total} ({100 * label_matches / total:.1f}%)")

    if all_score_diffs:
        print(f"\n  Score differences (across {len(all_score_diffs)} matched entities):")
        print(f"    Mean:   {np.mean(all_score_diffs):.10f}")
        print(f"    Max:    {np.max(all_score_diffs):.10f}")
        print(f"    Median: {np.median(all_score_diffs):.10f}")
        print(f"    StdDev: {np.std(all_score_diffs):.10f}")
        pct_zero = 100 * sum(1 for d in all_score_diffs if d == 0.0) / len(all_score_diffs)
        print(f"    Exact zero diff:  {pct_zero:.1f}% of entities")

    print("\n  Per-category breakdown:")
    for cat_name, stats in category_stats.items():
        print(f"    {cat_name:20s}: {stats['exact']}/{stats['total']} exact ({stats['pct']})")

    if mismatches:
        print(f"\n  Mismatches ({len(mismatches)}):")
        for m in mismatches[:15]:
            print(f"    [{m['index']}] ({m['category']}) \"{m['text'][:60]}\"")
            if m["torch_only"]:
                print(f"      Torch-only: {m['torch_only'][:3]}")
            if m["onnx_only"]:
                print(f"      ONNX-only:  {m['onnx_only'][:3]}")
            if m["max_score_diff"] is not None:
                print(f"      Max score diff: {m['max_score_diff']:.10f}")
        if len(mismatches) > 15:
            print(f"    ... and {len(mismatches) - 15} more")

    print(f"\n{'─' * 50}")
    print("INFERENCE SPEED (CPU)")
    print(f"{'─' * 50}")
    print(f"  Texts benchmarked:    {len(speed_texts)}")
    print("  Timed runs:           5")
    print()
    ts, os_ = torch_speed, onnx_speed
    print(f"  {'':20s}  {'Torch':>12s}  {'ONNX':>12s}  {'Δ':>10s}")
    print(f"  {'─' * 58}")
    print(f"  {'Total time (mean)':20s}  {ts['mean_total_s']:>10.3f}s"
          f"  {os_['mean_total_s']:>10.3f}s  {speedup:>8.2f}x")
    print(f"  {'Total time (std)':20s}  {ts['std_total_s']:>10.3f}s"
          f"  {os_['std_total_s']:>10.3f}s")
    print(f"  {'Per text (mean)':20s}  {ts['mean_per_text_ms']:>9.2f}ms"
          f"  {os_['mean_per_text_ms']:>9.2f}ms  {speedup:>8.2f}x")
    print(f"  {'Per text (p50)':20s}  {ts['p50_per_text_ms']:>9.2f}ms"
          f"  {os_['p50_per_text_ms']:>9.2f}ms")
    print(f"  {'Throughput':20s}  {ts['texts_per_second']:>8.1f}/s"
          f"  {os_['texts_per_second']:>8.1f}/s  {speedup:>8.2f}x")
    print(f"  {'Min total':20s}  {ts['min_total_s']:>10.3f}s"
          f"  {os_['min_total_s']:>10.3f}s")
    print(f"  {'Max total':20s}  {ts['max_total_s']:>10.3f}s"
          f"  {os_['max_total_s']:>10.3f}s")

    verdict = "FASTER" if speedup > 1.0 else "SLOWER"
    print(f"\n  ONNX is {abs(speedup):.2f}x {verdict} than Torch on CPU")

    print(f"\n{'=' * 90}")

    # Save detailed results
    results = {
        "accuracy": {
            "total_texts": total,
            "exact_match": exact_matches,
            "exact_match_pct": round(100 * exact_matches / total, 2),
            "label_match": label_matches,
            "label_match_pct": round(100 * label_matches / total, 2),
            "entity_count_torch": entity_count_torch,
            "entity_count_onnx": entity_count_onnx,
            "score_diff_mean": float(np.mean(all_score_diffs)) if all_score_diffs else None,
            "score_diff_max": float(np.max(all_score_diffs)) if all_score_diffs else None,
            "score_diff_median": float(np.median(all_score_diffs)) if all_score_diffs else None,
            "per_category": category_stats,
        },
        "speed": {
            "torch": {k: v for k, v in torch_speed.items() if k != "raw_timings"},
            "onnx": {k: v for k, v in onnx_speed.items() if k != "raw_timings"},
            "speedup": round(speedup, 3),
            "torch_timings": torch_speed["raw_timings"],
            "onnx_timings": onnx_speed["raw_timings"],
        },
        "mismatches": mismatches[:50],
    }

    output_path = "scripts/ab_test_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
