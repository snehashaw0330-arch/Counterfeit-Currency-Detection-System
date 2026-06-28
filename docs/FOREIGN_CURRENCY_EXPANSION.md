# Foreign Currency Expansion Notes

This note is for the next project phase: adding foreign currencies, polymer-note
handling, Guilloche verification, and a PUF-style note identity layer.

## Short answer

Yes, we can expand to other countries' notes.

The best practical split is:

- **Bangladesh (BDT)** for an immediate **foreign counterfeit dataset**
  because the repo already contains a fetch script for **JaalTaka**
  (`scripts/fetch_jaaltaka.py`).
- **Australia, Canada, UK, and the Philippines** for **polymer-note support**
  because they have strong official documentation for polymer security features.

## Important reality check

Bangladesh is a good **foreign-currency** target, but it should **not** be our
main polymer benchmark.

Why:

- The repo's existing Bangladesh script fetches **JaalTaka**, which is a set of
  **close-up smartphone crops** of genuine and physical counterfeit BDT notes.
- Those images are great for **security-region classification / forensic cues**.
- They are **not** full-note images, so they are a poor fit for:
  - note localization
  - whole-note proportion checks
  - full serial extraction
  - country detection from overall note layout

So Bangladesh should be treated as:

- **foreign counterfeit coverage**: yes
- **polymer-note benchmark**: no
- **full-note country-detection benchmark**: only after we gather whole-note BDT images

## Best countries to add first

### Tier 1: add now

1. **Bangladesh (BDT)**
   - Use for foreign-note/counterfeit coverage.
   - We already have `scripts/fetch_jaaltaka.py`.
   - Keep it in a separate country-aware pipeline.

2. **Canada (CAD)**
   - Excellent official polymer-security documentation.
   - Strong transparent-window features.
   - Good fit for polymer-specific feature engineering.

3. **United Kingdom (GBP)**
   - Official polymer-note feature pages are clear and detailed.
   - Includes see-through windows, hologram changes, raised print, tactile dots,
     and unique serial formatting.

4. **Australia (AUD)**
   - Very strong polymer reference set.
   - Official RBA guidance covers top-to-bottom window, 3D effects, moving bird,
     intaglio texture, and microprint.

### Tier 2: good follow-up

5. **Philippines (PHP)**
   - Official polymer series page is detailed and current.
   - Good modern polymer case with tactile dots, clear windows, vertical value
     panel, and serial-number placement.

## What to collect per country

For each currency, try to gather both:

- **full-note images**
  - obverse
  - reverse
  - flat scans
  - real phone photos
- **security-region crops**
  - serial area
  - window area
  - hologram / foil area
  - watermark area
  - tactile / intaglio area if visible

## Recommended dataset layout

Do not mix everything into the current INR dataset structure.

Use a country-aware structure like:

```text
dataset/
  foreign/
    bdt/
      full_note/
        real/
        fake/
      security_crops/
        real/
        fake/
    cad/
      full_note/
        real/
        fake/
    gbp/
      full_note/
        real/
        fake/
    aud/
      full_note/
        real/
        fake/
    php/
      full_note/
        real/
        fake/
```

Required sidecar metadata for each image:

```json
{
  "country": "Canada",
  "currency": "CAD",
  "substrate": "polymer",
  "denomination": "20",
  "side": "obverse",
  "label": "real",
  "source": "Bank of Canada official page image"
}
```

Repo helpers for this layout:

- `scripts/fetch_polymer_reference_images.py`
- `scripts/validate_foreign_dataset.py`
- `scripts/benchmark_foreign_modules.py`

Also store metadata per image:

- `country`
- `currency`
- `substrate` = `paper | polymer`
- `denomination`
- `side` = `obverse | reverse | crop`
- `label` = `real | fake`
- `source`
- `feature_region` if crop

## How this maps to your new requirements

### 1. Guilloche pattern generator from serial number

This generalizes well across countries.

- Input: serial string
- Normalize: uppercase, remove separators that are not meaningful
- Seed a deterministic generator from the normalized serial
- Same serial -> same generated guilloche

This should be **currency-agnostic**.

### 2. Guilloche pattern verification

This will be easier on currencies where:

- the guilloche region is stable
- note templates are consistent
- we have full-note images or aligned crops

Best first approach:

- implement **template-specific region extraction** per currency
- compare:
  - generated serial-seeded pattern
  - extracted real-note pattern region

### 3. Digital Physical Unclonable Function (PUF)

For software-only work, treat PUF as a **digital proxy**, not a hardware claim.

A practical project version is:

- extract a stable micro-texture signature from a note region
- hash / embed it as the note's physical fingerprint
- compare re-captured signatures for consistency

Best candidate regions:

- polymer window edge textures
- intaglio / raised-print regions
- microprint regions
- fine-line guilloche regions

### 4. Add support for foreign polymer banknotes

This is where **Canada, UK, Australia, Philippines** help most.

Polymer-specific checks we can add:

- clear-window presence and geometry
- window transparency consistency
- foil / hologram color-shift proxy
- raised-print / intaglio texture proxy
- microprint sharpness
- tactile-dot region presence
- substrate gloss / reflectance cues

### 5. Automatic country and currency detection

Do this in two stages:

1. **Country detection**
   - layout classifier
   - color/layout embeddings
   - OCR text cues like `Bank of England`, `Canada`, denomination words, scripts

2. **Currency / denomination detection**
   - country-specific denomination classifier
   - denomination OCR fallback

This is much safer than trying to classify all denominations across all
currencies in one flat label space from day one.

### 6. Multi-currency counterfeit detection

Recommended architecture:

- Stage A: detect `country`
- Stage B: route to country-specific forensic checks
- Stage C: combine with a substrate-aware counterfeit model

That avoids mixing:

- INR paper-note rules
- polymer-note rules
- close-up-crop datasets like JaalTaka

## Bangladesh-specific note for the team

Bangladesh is still very worth adding because:

- we already have a repo script for it
- JaalTaka contains **genuine vs physical counterfeit** imagery
- it gives us a real foreign-counterfeit branch early

But for the report and demos, phrase it carefully:

- **Indian vs Bangladesh** comparison: acceptable
- **paper vs polymer** comparison using Bangladesh: not ideal

If we want a clean polymer comparison in the report, prefer:

- **India (paper) vs Canada/UK/Australia/Philippines (polymer)**

## Suggested project wording

If you need one line for the report or viva:

> We extended the system toward multi-currency support by separating
> country-aware detection from substrate-aware forensic verification, using
> Bangladesh for foreign counterfeit coverage and polymer-banknote countries
> such as Canada, the UK, Australia, and the Philippines for polymer-specific
> security analysis.

## Sources

These are the sources I checked while preparing this note.

- Repo script for Bangladesh dataset:
  - `scripts/fetch_jaaltaka.py`
- Bank of Canada, `$20 polymer note security features`:
  - https://www.bankofcanada.ca/banknotes/20-polymer-note-security-features/
- Bank of England, `£20 note`:
  - https://www.bankofengland.co.uk/banknotes/polymer-20-pound-note
- Bank of England, `£50 note`:
  - https://www.bankofengland.co.uk/banknotes/polymer-50-pound-note
- Reserve Bank of Australia, `List of Security Features`:
  - https://banknotes.rba.gov.au/counterfeit-detection/list-of-security-features/
- Bangko Sentral ng Pilipinas, `Polymer Banknotes`:
  - https://www.bsp.gov.ph/SitePages/CoinsAndNotes/PolymerBanknote.aspx

## Recommended next build order

1. Keep Bangladesh as a separate foreign dataset branch.
2. Add one polymer full-note currency first: **Canada or UK**.
3. Make the dataset index country-aware and substrate-aware.
4. Implement country detection before multi-currency counterfeit scoring.
5. Add polymer-specific forensic checks only after the routing layer exists.
