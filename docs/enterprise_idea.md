# Enterprise Venue AI Music Platform
### A Karpathy-Style Research Idea Document

> **Author:** Yusuf Kerim Sarıtaş  
> **Advisor:** Dr. Öğr. Üyesi Nurettin Şenyer  
> **Date:** June 2026  
> **Status:** Early-stage research hypothesis

---

## 0. The One-Sentence Pitch

> An AI system that generates royalty-free, harmonically coherent background music sets — personalized in real-time for any commercial venue by venue type, customer demographics, time of day, and brand identity — eliminating licensing costs while delivering DJ-quality audio atmosphere.

---

## 1. Problem Statement

### 1.1 The DJ Friend Problem

A DJ friend of mine recently played a paid set at a venue. The next day, a shopping mall like LCW or DeFacto is playing generic, repetitive background music from a third-party service — paying thousands of lira per month for something that sounds worse than a free Spotify playlist. Meanwhile, cafes, restaurants, hospitals, and corporate offices all need background music but face a brutal tradeoff:

**Option A:** Use Spotify/YouTube → illegal for commercial use in Turkey (Law 5846), risk of MESAM/MÜ-YAP fines  
**Option B:** Pay for licensed background music services → expensive, generic, no customization  
**Option C:** Stay silent → terrible customer experience, measurably reduces dwell time and sales

None of these options are good. There is a better fourth option that does not exist yet.

### 1.2 What Is Actually Missing

Current background music services solve *licensing* but ignore *quality*. They play legally safe, mood-appropriate music — but:

- Transitions between tracks are abrupt (jarring key changes, BPM jumps)
- No adaptation to energy curve throughout the day (morning open → afternoon busy → evening close)
- No harmonic coherence — the sequence sounds like a shuffled playlist, not a curated set
- No demographic adaptation — a 50+ customer in a pharmacy gets the same playlist as a 20-year-old in a streetwear store
- Music is often AI-generated with a "synthetic" feel that subliminally lowers perceived brand quality

The gap: **nobody applies DJ-grade harmonic curation to commercial venue music.**

### 1.3 Why Now

Three forces converge in 2025–2026:

1. **Generative AI for music matured.** Meta MusicGen, Suno v3, Udio, and open-source equivalents can now produce genre/mood/tempo-specific audio indistinguishable from produced tracks at scale.
2. **Copyright enforcement tightened.** MESAM and MÜ-YAP have significantly increased inspections of commercial venues in Turkey. The latent fear is now acute.
3. **TrackBang's existing infrastructure.** A working key/BPM detection model (47.7% accuracy, 24-class), Camelot compatibility engine, and production backend already exist. The marginal cost of building this is dramatically lower than a greenfield startup.

---

## 2. Proposed Solution

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    VENUE PROFILE                             │
│  venue_type | age_range | theme | hour | occupancy | season │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              SET PLANNING ENGINE (LLM + Rules)               │
│                                                             │
│  Input:  venue profile + energy curve target                 │
│  Output: [ {key, bpm, mood, duration}, ... ] × N tracks     │
│                                                             │
│  Constraints:                                               │
│    - Camelot wheel: adjacent keys only (±1 step)            │
│    - BPM delta: max ±8 BPM between consecutive tracks       │
│    - Energy arc: cosine curve mapped to open→close hours    │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┴──────────────┐
            │                            │
            ▼                            ▼
┌───────────────────┐        ┌───────────────────────┐
│   MUSIC LIBRARY   │        │   AI MUSIC GENERATOR  │
│   (owned tracks)  │        │   (MusicGen / Suno)   │
│                   │        │                       │
│  - Genre tagged   │        │  Prompt: "ambient     │
│  - Key/BPM known  │        │  lounge jazz, 92 BPM, │
│  - Mood scored    │        │  A minor, 3 min"      │
│  - Owned rights   │        │                       │
└────────┬──────────┘        └──────────┬────────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              AUDIO POST-PROCESSING                         │
│                                                           │
│  - EQ normalization across tracks                         │
│  - Crossfade transition (harmonic mix, 4–8 bar overlap)   │
│  - Loudness normalization (EBU R128 / -14 LUFS)           │
│  - Optional: brand jingle insertion at configurable slots  │
└───────────────────────────────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────────┐
│              DELIVERY                                      │
│                                                           │
│  - Web Player (browser-based, zero hardware)              │
│  - Smart Stream (adaptive bitrate)                        │
│  - Offline Radio Player (continues if internet drops)     │
│  - REST API (for POS system integration, IoT sensors)     │
└───────────────────────────────────────────────────────────┘
```

### 2.2 The Camelot Harmonic Engine (Core Differentiator)

The Camelot Wheel maps 24 musical keys to a clock-face grid. Adjacent keys (±1 position) are harmonically compatible — transitions between them are smooth to the human ear.

```
        12B  1B  2B
    11B           3B
  10B               4B
   9B               5B
    8B           6B
        7B  6B  5B   ← inner ring = minor keys (A)
```

**Current state (TrackBang v4):** The key detection model classifies any 30-second audio clip with 47.7% top-1 accuracy. This is sufficient for transition planning — even with errors, Camelot adjacency constraints prevent harmonically dissonant transitions.

**For venue use:** Instead of detecting keys of uploaded tracks, the system *generates* tracks with the desired key. This eliminates detection error entirely for the core harmonic pipeline.

### 2.3 Venue Profile Schema

```json
{
  "venue": {
    "type": "cafe | restaurant | retail | hotel_lobby | spa | gym | hospital | pharmacy | office",
    "brand_tier": "budget | mid | premium | luxury",
    "primary_age_range": "18-25 | 25-35 | 35-50 | 50+",
    "theme": "string (e.g. 'industrial minimalist', 'mediterranean', 'streetwear')"
  },
  "schedule": {
    "open": "09:00",
    "close": "22:00",
    "peak_hours": ["12:00-14:00", "18:00-20:00"],
    "energy_curve": "morning_calm | constant | rising | peak_evening"
  },
  "music_preferences": {
    "genres": ["lounge", "deep_house", "acoustic_pop"],
    "exclude_genres": ["metal", "explicit_rap"],
    "bpm_range": [80, 120],
    "language_preference": "instrumental | turkish | english | mixed"
  },
  "branding": {
    "jingle_enabled": true,
    "announcement_slots": ["12:00", "18:00"],
    "custom_voice_id": "string"
  }
}
```

### 2.4 Energy Curve Mapping

```
Energy
  1.0 │                              ╭──╮
  0.9 │                         ╭───╯  ╰──╮
  0.8 │                    ╭───╯          ╰──
  0.7 │               ╭───╯
  0.6 │          ╭───╯
  0.5 │     ╭───╯
  0.4 │╭───╯
  0.3 ╰┤
      └──────────────────────────────────────── Time
      09:00  11:00  13:00  15:00  17:00  19:00  21:00

Energy = f(BPM, key_mode, production_density)
  BPM:     80 BPM (open) → 128 BPM (peak) → 95 BPM (close)
  Mode:    Major (energetic) ↔ Minor (intimate/dark)
  Density: sparse acoustic → full electronic
```

---

## 3. Market Analysis

### 3.1 Global Market Size

| Segment | 2025 Value | 2031 Projection | CAGR |
|---------|-----------|----------------|------|
| Background Music (Global) | $3.0B | $5.1B | 9.6% |
| Commercial Background Music | $2.04B | $2.78B | 6.35% |
| AI in Music (Broader) | $4.48B | $8.5B | 23.7% |

Source: Polaris Market Research, Mordor Intelligence, Sound Verse AI (2025)

### 3.2 Turkish Market Sizing (Bottom-Up Estimate)

```
Cafes & Restaurants in Turkey:    ~350,000
Retail Stores (chain):            ~85,000
Hotels (3-5 star):                ~4,500
Hospitals & Clinics:              ~35,000
Corporate Offices (50+ person):   ~120,000
─────────────────────────────────────────
Total Addressable Venues:         ~594,500

Realistic penetration (Y3):       1.5%  →  ~8,900 venues
Average ARPU:                     ₺500/month
Annual Revenue Potential (Y3):    ₺53M/year (~$1.6M USD)
```

**Note:** These are conservative estimates. SMG currently serves 13,000+ Turkish locations. Capturing 50% of that scale would yield ₺78M/year.

### 3.3 Customer Segmentation

**Segment 1 — SMB Venues (primary beachhead)**
- Independent cafes, restaurants, barbershops
- Pain: Spotify is illegal but they use it anyway; fear of MESAM fines
- Willingness to pay: ₺300–600/month
- Decision maker: owner (single person)
- Key message: "Legal, affordable, sounds better than Spotify"

**Segment 2 — Retail Chains (scale play)**
- LCW, DeFacto, Zara TR, Boyner, Mediamarkt TR
- Pain: centralized control across 100s of stores, brand consistency
- Willingness to pay: ₺200–400/store/month (×hundreds of stores)
- Decision maker: marketing director
- Key message: "One dashboard, 500 stores, brand-consistent audio"

**Segment 3 — Premium Hospitality (premium positioning)**
- 5-star hotels, fine dining, luxury spas
- Pain: current solutions feel generic for their brand tier
- Willingness to pay: ₺2,000–8,000/month
- Decision maker: F&B director or brand manager
- Key message: "Your own radio station. Sounds like your brand."

**Segment 4 — Healthcare (long-term, underserved)**
- Hospitals, clinics, pharmacies, waiting rooms
- Pain: generic "hospital music" lowers perceived care quality
- Research backing: 2023 NIH study confirms music reduces patient anxiety by 37%
- Willingness to pay: ₺800–2,000/month
- Decision maker: hospital administrator
- Key message: "Clinically-informed music reduces patient stress"

---

## 4. Competitive Analysis

### 4.1 Global Competitors

| Company | Founded | Scale | AI? | Harmonic Curation? | Weakness |
|---------|---------|-------|-----|-------------------|----------|
| **Mood Media** (former Muzak) | 1934 | 500K+ locations, 140 countries | Limited | No | Legacy tech, expensive, no Turkey office |
| **Soundtrack Your Brand** | 2013 | Active in Turkey | Playlist AI (2024) | No | Spotify-dependent, licensing costs passed to customer |
| **Brandtrack** | 2015 | McDonald's, Hilton | AI + human hybrid | No | Premium pricing, not SMB-friendly |
| **Rockbot** | 2011 | 1M+ users | Recommendation AI | No | US-focused, no Turkish support |
| **PlayNetwork** | 1996 | Starbucks partner | No | No | Extremely enterprise, inaccessible to SMB |

**Key global insight:** Not a single major player applies harmonic compatibility (Camelot wheel) to background music scheduling. They all treat tracks as independent units, not as a harmonic sequence.

### 4.2 Turkish Competitors

| Company | Scale | AI? | Harmonic Curation? | Core Weakness |
|---------|-------|-----|-------------------|--------------|
| **SMG** | 13,000+ TR locations | Basic scheduling | No | Music is outsourced, not generated; generic |
| **Mağaza Radyo** | 5,700 locations | No | No | Pure playlist service, no tech differentiation |
| **Kurumsal Müzik Net** | Unknown | AI generation ✓ | No | Generates music but no curation intelligence |
| **VIVA MEDYA** | Small | No | No | Corporate radio focus, not venue-specific |
| **NOT FM / MAGAZA FM** | Small | No | No | Genre-based only |

**Key Turkish insight:** Kurumsal Müzik Net is the closest competitor — they solve the licensing problem through AI generation. But they have no harmonic curation layer. Their product is "legal music" not "intelligently sequenced legal music."

### 4.3 Competitive Positioning Map

```
                    HIGH HARMONIC QUALITY
                           │
                    TARGET │◀── Our position
                           │
    GENERIC ───────────────┼─────────────────── PERSONALIZED
    MUSIC                  │                         MUSIC
                           │
    Kurumsal  SMG    Mağaza│          Soundtrack   Brandtrack
    Müzik Net        Radyo │          Your Brand
                           │
                    LOW HARMONIC QUALITY
```

### 4.4 Sustainable Competitive Advantages

1. **Proprietary harmonic curation engine** — 2+ years ahead of any Turkish competitor to build
2. **Owned music library** — grows with every generation; competitors cannot replicate without years of production
3. **Key detection infrastructure** — already built (TrackBang v4 model); competitors would need to build from scratch
4. **Data flywheel** — more venues → more behavioral data → better personalization → more venues
5. **Turkish market relationship** — local language, local support, understanding of Turkish music culture

---

## 5. SWOT Analysis

### Strengths
- **Working key/BPM detection model** (858K params, 47.7% val accuracy, production-deployed)
- **Camelot harmonic algorithm** already in production (TrackBang CUE AI)
- **FastAPI infrastructure** — the key detection API is running; extension is incremental
- **Founder is a DJ** — 7+ years domain expertise; can evaluate output quality by ear
- **Zero external licensing** — AI-generated music model eliminates MESAM/MÜ-YAP dependency
- **First-mover advantage** in harmonically-aware venue music in Turkey

### Weaknesses
- **AI-generated music quality gap** — current generative models (MusicGen, Suno) produce good but not excellent music; some venues may perceive lower brand quality vs. real artists
- **Cold start problem** — need to generate a large, diverse music library before product is sellable
- **Sales is hard** — B2B SaaS selling to Turkish SMBs requires boots-on-ground sales; not a pure product-led growth play
- **No brand recognition** in B2B space — TrackBang is known in DJ community, not in retail/hospitality
- **Hardware/offline requirement** — venues need offline playback; requires embedded software or hardware device

### Opportunities
- **MESAM enforcement wave** — increasing fines are creating acute pain for the 90%+ of Turkish venues using Spotify illegally
- **Healthcare vertical** — completely unserved; no competitor has a product here
- **International expansion** — Turkish-language advantage in MENA/Turkic-speaking markets (Azerbaijan, Kazakhstan, Uzbekistan)
- **API/white-label play** — license the harmonic engine to existing services (Mağaza Radyo, SMG) who lack tech but have distribution
- **Generative AI improving rapidly** — music quality will reach parity with produced tracks within 2–3 years

### Threats
- **Mood Media entering Turkey seriously** — they have capital and relationships; if they build harmonic curation, advantage shrinks
- **Spotify launching commercial tier in Turkey** — Spotify already offers commercial licensing in some markets; if they expand to Turkey with AI playlist curation, SMB segment becomes harder
- **AI music copyright law changes** — if AI-generated music becomes subject to collective licensing (currently unsettled globally), the core cost advantage disappears
- **Suno/Udio offering direct B2B** — if generative AI music companies build venue scheduling on top of their own generation, they could bypass the middle layer

---

## 6. Business Model

### 6.1 Revenue Streams

**Primary: SaaS Subscription**
```
Tier        | Venues | Price/month | Features
────────────|────────|─────────────|──────────────────────────────────
Starter     | 1      | ₺399        | Basic scheduling, 3 venue profiles
Professional| 1-5    | ₺799        | + Announcements, analytics, offline
Business    | 5-50   | ₺299/venue  | + API access, custom voice, priority
Enterprise  | 50+    | Custom      | + White-label, dedicated support
```

**Secondary: Hardware Device (optional)**
- Raspberry Pi-based "TrackBang Box" — plug-and-play offline player
- One-time ₺1,500 hardware + ₺199/month subscription
- Eliminates IT complexity for non-technical venue owners

**Tertiary: API Licensing**
- License the harmonic scheduling engine to competing services
- ₺0.001/request or revenue share model

### 6.2 Unit Economics (Conservative Y2 Estimate)

```
Customer Acquisition Cost (CAC):    ₺800  (field sales + digital)
Average Revenue Per User (ARPU):    ₺550/month
Gross Margin:                       ~78%  (cloud + generation costs ~22%)
Payback Period:                     ~1.5 months
LTV (24-month churn 3%/month):      ₺10,200
LTV/CAC:                            12.75x  ← strong unit economics
```

### 6.3 Go-to-Market Strategy

**Phase 1 — Beachhead (Months 1–6)**
- Target: 50 independent cafes in İstanbul (Kadıköy, Beyoğlu, Nişantaşı)
- Channel: Direct outreach, DJ community referrals, TrackBang user base
- Offer: 3 months free, then ₺399/month
- Goal: Product-market fit signal, 10 paying customers by month 6

**Phase 2 — Scale (Months 7–18)**
- Target: Istanbul + Ankara + İzmir, SMB segment
- Channel: Digital marketing (Google Ads targeting "MESAM ceza"), partnerships with Yemeksepeti/GetirYemek restaurant networks
- Goal: 500 paying venues, ₺275K MRR

**Phase 3 — Enterprise (Months 18–36)**
- Target: Retail chains (approach marketing directors)
- Channel: LinkedIn outreach, trade shows (RetailTurkey, HoReCa Istanbul)
- Goal: 3 enterprise contracts, 2,000+ total venues

---

## 7. Technical Research Questions

These are the open problems that need to be solved. I do not know the answers yet.

**Q1: How much does harmonic coherence actually improve venue KPIs?**
- Hypothesis: Harmonically coherent background music increases dwell time by 8–15% and reduces subliminal discomfort
- How to test: A/B test two cafes with identical venue profiles but one gets Camelot-sequenced music, one gets random playlist
- Measurement: POS data (average check size), time-stamped entry/exit counting (camera + CV), staff subjective rating

**Q2: What is the minimum music library size for a viable product?**
- Hypothesis: 500 tracks per genre (10 genres = 5,000 tracks) is enough for 30-day non-repetitive coverage
- Critical: A customer hearing the same track twice in a week destroys trust
- How to test: Calculate entropy of playlist generation given library size constraints

**Q3: Can MusicGen/Suno produce venue-ready audio at sufficient quality?**
- Known limitation: Current open-source generative models occasionally produce artifacts (glitches, tempo drift, unnatural harmony changes) at ~3–5% of outputs
- How to test: Blind A/B test with 100 venue owners: "Which audio feels more professional?"
- Acceptable threshold: <1% artifact rate for production use

**Q4: How should the Camelot engine handle AI-generated tracks with imperfect key center?**
- AI-generated music sometimes has ambiguous key center (especially ambient/atonal tracks)
- The existing v4 key detection model (47.7% accuracy) was trained on electronic music — will it generalize to AI-generated ambient?
- How to test: Generate 500 tracks with MusicGen, detect with v4 model, manually verify — measure precision on known-key prompts

**Q5: What is the optimal transition architecture for seamless crossfading?**
- Option A: Fixed-length crossfade (8 bars at 4/4 = 16 seconds)
- Option B: Beat-synchronized crossfade (requires BPM-accurate beat detection)
- Option C: AI-generated transition segment (generate a 4-bar bridge between two tracks)
- Option C is most novel but unproven at scale

**Q6: How do we handle non-Western scales and Turkish classical music requests?**
- Camelot wheel assumes Western 12-tone equal temperament
- Turkish maqam music uses microtonal scales — Camelot is inapplicable
- Opportunity: Build a Maqam-compatible sequencing system (completely novel, no prior work)

---

## 8. Implementation Roadmap

### MVP (3 months) — Prove the core loop works

```
Week 1-2:   MusicGen integration — generate 100 test tracks by prompt
Week 3-4:   Key detection validation on generated tracks
Week 5-6:   Camelot sequencer — given venue profile, output ordered track list
Week 7-8:   Audio crossfading pipeline — seamless transitions
Week 9-10:  Web player — browser-based, no app install required
Week 11-12: 5 beta cafe pilots in Istanbul
```

**MVP success criteria:**
- [ ] 1-hour set generated end-to-end from venue profile in < 2 minutes
- [ ] Zero audibly jarring transitions in 10-listen user test
- [ ] 5 cafe owners willing to pay after free trial

### V1 Product (6 months)
- Venue dashboard (React)
- Scheduling (weekly program, energy curve editor)
- Announcement integration (TTS with custom voice)
- Offline player (Electron app or Raspberry Pi image)
- 10 genre × 500 track library

### V2 Product (12 months)
- Real-time occupancy adaptation (IoT sensor or camera integration)
- Analytics (dwell time correlation, customer feedback)
- Mobile app for venue owners
- API for POS integration
- Healthcare vertical pilot

---

## 9. Why This Will Be Hard

I want to be honest about the difficulties because optimism without realism is useless.

1. **Music quality is a feel, not a metric.** A cafe owner who "just doesn't like the music" will churn, even if all KPIs are good. Subjective quality is hard to A/B test and hard to improve systematically.

2. **B2B SMB sales in Turkey is relationship-driven.** Turkish SMB owners buy from people they trust, not from websites. This requires a sales team on the ground. That is expensive and slow.

3. **The product is ambient, not exciting.** Nobody opens a cafe and tweets about their background music service. This is a silent, invisible utility — which means word-of-mouth is slow and PR is hard.

4. **Generative AI music is improving fast, but so are the giants.** Spotify, YouTube Music, and Apple Music are all investing heavily in AI. If Spotify launches a commercial tier in Turkey with AI scheduling, they have infinite music library and global brand trust.

5. **Integration hell.** Different venues have different audio systems, internet quality, and technical sophistication. Building reliable offline playback that "just works" in a Turkish cafe (with inconsistent Wi-Fi and a non-technical owner) is a genuine engineering challenge.

---

## 10. Prior Work and References

### Academic
- Tzanetakis, G. & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Trans. Speech Audio Process.* — foundational work on audio feature extraction
- Müller, M. (2015). *Fundamentals of Music Processing.* Springer. — covers Chroma features, HPSS, key detection
- Bitteur, H. et al. (2020). Computational analysis of harmonic compatibility in DJ mixing. — closest academic precedent for Camelot-style analysis
- Li, T. et al. (2023). Context-aware music recommendation for commercial environments. *ISMIR 2023.* — venue-aware recommendation
- Schneider, S. et al. (2023). Music in healthcare settings: systematic review of clinical outcomes. *NIH PMC10380075.* — evidence base for healthcare vertical

### Industry
- Soundtrack Your Brand (2024). AI Playlist Generator announcement. PR Newswire.
- Mood Media. (2025). Global background music market presence. moodmedia.com
- SMG. (2025). Music for Business. smg.com.tr
- Kurumsal Müzik Net. (2025). AI-Destekli Müzik Üretimi. kurumsalmuzik.net
- Copet, J. et al. (2023). Simple and Controllable Music Generation (MusicGen). *NeurIPS 2023.* arXiv:2306.05284

### Market Data
- Polaris Market Research. (2025). Background Music Market Size Report 2025–2034.
- Mordor Intelligence. (2026). Commercial Background Music Market Report.
- Sound Verse AI. (2025). AI Music Industry Trends 2026.

---

## 11. Open Questions for the Advisor

1. Is there prior academic work specifically on **harmonic sequence optimization for ambient/background music** (as opposed to DJ mixing)? If not, this may be a publishable contribution.

2. Should the key detection model (v4) be fine-tuned on AI-generated music, or is the distribution gap small enough to ignore?

3. Is the **Turkish maqam extension** (Q6 above) a viable thesis-level contribution? It would be entirely novel.

4. For the A/B testing methodology (Q1), is there a standard protocol in music psychology literature for measuring dwell time effects?

---

*This document is a living research note. Last updated: June 2026.*  
*Next review: When MVP is deployed with first 5 beta venues.*
