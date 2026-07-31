# Oralable system map — Mermaid diagrams (pitch / Notion)

**App working diagrams:** [MOBILE_APP_FLOWS.md §2](../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0)

**As at:** 27 Jul 2026 · Pair with interactive canvas + [ORALABLE_SYSTEM_MAP.csv](./ORALABLE_SYSTEM_MAP.csv)  
**Timeline truth:** [PRODUCT_ROADMAP.md §3](./PRODUCT_ROADMAP.md#3-timeline-calendar--canonical) · avenues [§2b](./PRODUCT_ROADMAP.md#2b-technology-avenues)  
**Raster / photo placeholders:** [FIGURES.md](./FIGURES.md) (do not convert these Mermaid blocks to PNG)

Paste any block into GitHub, Notion, or a Mermaid-capable slide tool.

---

## 1. Development timeline

```mermaid
flowchart LR
  A[2024-25 Gen1+IP] --> B[H1'26 kits FW 1.0.70]
  B --> C[24 Jul eng: Protocol A + night PDF]
  C --> D[Now-Sep'26 Phase 0 YOU ARE HERE]
  D --> E[Q4'26-Q1'27 Phase 1+]
  E --> F[Gen2 parallel]
  E --> G[H2'27-28 Stage B]
```

---

## 2. System stack — live vs deferred

```mermaid
flowchart TB
  subgraph HW[Hardware Gen1]
    CLIP[Clip PPG+ACC]
    CASE[Magnetic case]
  end
  subgraph FW[Firmware]
    NRF[nRF52832 TGM GATT]
  end
  subgraph APP[Software]
    IOS[Patient iOS 4.3.3]
    CORE[OralableCore]
    PDF[Night PDF / CSV]
  end
  subgraph LATER[Deferred]
    PRO[Dentist app]
    CK[CloudKit]
    SB[Stage B]
  end
  CLIP --> NRF
  CASE --> NRF
  NRF --> IOS
  IOS --> CORE
  CORE --> PDF
  PDF -.-> PRO
  IOS -.-> CK
  CK -.-> PRO
  PDF -.-> SB
```

---

## 3. GTM phase gates

```mermaid
flowchart TB
  P0[Phase 0 temple vitals] --> PATHA[Path A Consumer]
  P0 --> GATE[Phase 0 pass]
  GATE --> P1[Phase 1+ muscle UX]
  P1 --> PREM[Soft Premium optional]
  P1 --> PATHB[Path B Dentist + CloudKit]
  PATHB --> STAGEB[Stage B medical later]
```

---

## 4. Evidence protocols

```mermaid
flowchart LR
  A[Protocol A minutes] --> ML[Core ML gold / Tier 0-1]
  B[Protocol B structured] --> P1[Phase 1+ muscle evidence]
  O[Overnight ≥6 h] --> NR[Night report bands + hypnogram]
  ML --> APP[Patient app models]
  P1 --> APP
  NR --> APP
  NR -.-> PRO[Dentist share later]
```

---

## 5. Ken diligence readiness (conceptual)

```mermaid
pie title Ken areas by status count
  "Partial" : 8
  "Gap" : 3
```

---

## 6. Point B funding stack

```mermaid
flowchart LR
  FF[F&F €50k TBD] --> PB[Point B €180k by Oct 2026]
  CLN[PSSF CLN €100k TBD] --> PB
  HPSU[HPSU €30k TBD] --> PB
  PB --> STAGEA[Stage A Phase 0→1+ runway]
```

---

## 7. Modality ladder (where Oralable sits)

```mermaid
flowchart TB
  subgraph gold [Clinical_gold_standards]
    MEP[MEP_Dwave]
    EMG[sEMG]
    EEG[EEG_CMC]
  end
  subgraph optical [Optical_hemodynamic]
    OMG[Temple_IR_DC_OMG]
    FNIRS[Scalp_fNIRS]
    Vitals[Temple_HR_SpO2]
  end
  subgraph oralable [Oralable_Stage_A]
    P0[Phase0_vitals]
    P1[Phase1_bruxism_SASHB]
  end
  MEP -.->|orthogonal| OMG
  EEG -.->|ANS_HRV_proxy_only| Vitals
  EMG -->|validation_adjacent| OMG
  OMG --> P1
  Vitals --> P0
  P0 --> P1
  FNIRS -.->|related_physics| OMG
```

Source: [GEMINI_TEMPLE_PPG_AVENUES.md](./data_room/GEMINI_TEMPLE_PPG_AVENUES.md) · landscape §4b.

---

## 8. Patient app working model (Phase 0)

Canonical detail + more diagrams: [oralable_swift/docs/MOBILE_APP_FLOWS.md §2](../../oralable_swift/docs/MOBILE_APP_FLOWS.md#2-how-the-patient-app-works--phase-0) · placeholders [FIGURES.md](./FIGURES.md) (`FIG-IOS-006`…`008`).

```mermaid
flowchart LR
  Charge[Charge on case] --> Pair[Pair BLE]
  Pair --> Place[Temple placement]
  Place --> Stream[Worn-gated 50Hz]
  Stream --> UI[Dashboard HR SpO2]
  Stream --> Auto[Auto-record]
  Auto --> Pack[CSV plus clinical PDF]
```

```mermaid
flowchart TB
  Main[MainTabView]
  Main --> Dash[Dashboard]
  Main --> Dev[Devices]
  Main --> Sh[Share]
  Main --> Set[Settings]
```