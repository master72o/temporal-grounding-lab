# Temporal Grounding Benchmark Dataset Card

## Dataset Summary
The **Temporal Grounding Benchmark** is a specialized multimodal benchmark designed to evaluate temporal event localization, timestamp precision, and temporal hallucination rates in Vision-Language Models (Video-LLMs) and Multimodal Foundation Models.

- **Total Scenarios**: 25 curated video episodes
- **Total Ground Truth Events**: 80+ granular temporal events
- **Duration Range**: 20.0 seconds to 300.0 seconds (short, medium, long)
- **Modalities**: Visual (`video`), Acoustic (`audio`), Cross-Modal Synchronized (`audiovisual`)

---

## Domain Taxonomy

| Domain | Scenarios | Representative Tasks | Primary Modality Cues |
| :--- | :--- | :--- | :--- |
| **`Surveillance_Security`** | 5 | Perimeter breaches, unauthorized entry, unattended baggage drop, window shatter | Wire cutting clangs, door latches, flashlight sweeps |
| **`Autonomous_Driving`** | 5 | Crosswalk pedestrians in fog, emergency sirens, lane debris swerving, cyclist blindspots | Doppler sirens, brake friction chirp, bike bells |
| **`Medical_Surgical`** | 5 | Trocar insertion, robotic suture knot tying, polyp electrocautery, phacoemulsification | Insufflation hiss, cautery buzz, ultrasound whine |
| **`Sports_Athletics`** | 5 | Fast break buzzer beaters, penalty kick saves, tennis Hawk-Eye reviews, 100m sprint starts | Whistles, starting pistol gunshots, bat cracks |
| **`Podcast_Dialogue`** | 5 | Multi-speaker debate cross-talk, slide deck transitions, guest mic mutes, hardware unboxing | Acoustic overlapping waveforms, applause, packaging tears |

---

## Evaluation Metric Reference

1. **Temporal Intersection-over-Union (tIoU)**:
   $$\text{tIoU} = \frac{|(s_{\text{pred}}, e_{\text{pred}}) \cap (s_{\text{gt}}, e_{\text{gt}})|}{|(s_{\text{pred}}, e_{\text{pred}}) \cup (s_{\text{gt}}, e_{\text{gt}})|}$$
2. **Recall@1 at IoU Thresholds ($R@1, \text{IoU} \ge \tau$)**:
   Percentage of ground truth events whose top predicted interval achieves $\text{tIoU} \ge \tau$ ($\tau \in \{0.3, 0.5, 0.7\}$).
3. **Boundary Jitter Error (BJE) in Seconds**:
   Mean absolute error across start and end boundaries:
   $$\text{BJE} = \frac{|s_{\text{pred}} - s_{\text{gt}}| + |e_{\text{pred}} - e_{\text{gt}}|}{2}$$
4. **Temporal Hallucination Rate (THR)**:
   Percentage of predicted intervals with $\text{tIoU} = 0.0$ or exceeding video duration bounds.
5. **Temporal Ordering Accuracy (TOA)**:
   Pairwise concordance of predicted start timestamps with ground truth sequential causality.

---

## Schema Example

```json
{
  "video_id": "SURV-01",
  "domain": "Surveillance_Security",
  "title": "Perimeter Fence Breach Incident",
  "duration_sec": 120.0,
  "fps": 30.0,
  "query_prompt": "Locate the moments when the trespasser approaches the fence, cuts the wire, and enters the secure compound.",
  "ground_truth_events": [
    {
      "event_id": "SURV-01-E1",
      "label": "Suspect Approaching Fence",
      "start_sec": 14.5,
      "end_sec": 26.0,
      "description": "Individual in dark hoodie walks from tree line toward north perimeter fence",
      "modality": "video"
    }
  ]
}
```
