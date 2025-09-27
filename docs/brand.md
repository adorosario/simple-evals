# CustomGPT.ai — Brand Kit v1 (Draft)

> A practical, developer‑ready brand system derived from the current website visuals. Treat this as a living doc—update as product/positioning evolves.

---

## 1) Brand Essence

**Positioning**  
Enterprise‑grade, no‑code AI agents built from your content. Accuracy‑first, secure, and fast to launch.

**Value pillars**  
1) **Accuracy you can trust** (anti‑hallucination, citations)  
2) **Enterprise‑ready** (SOC‑2, GDPR, encryption)  
3) **No‑code speed** (minutes to deploy; 100+ integrations)  
4) **Revenue & efficiency** (reduce tickets, faster internal answers)

**Voice & tone**  
Confident, helpful, technically credible, never hyped. Prefer plain English over jargon. Show, don’t overclaim. Avoid fear‑based messaging.

---

## 2) Logo System

**Primary lockup**  
Wordmark: “Custom” in blue, “GPT.ai” in pink. Brain‑chip symbol with purple→pink gradient.

**Clearspace**  
Minimum clearspace = height of the “C”. Keep free of text/graphics.

**Minimum size**  
Digital: **128px** width. Print: **30mm** width.

**Backgrounds**  
- Light backgrounds: use full‑color gradient mark + wordmark.  
- Dark/photography: use **reversed** (white wordmark) with gradient mark or a **mono** white mark.

**Don’ts**  
✗ No stretching or skewing  
✗ No drop‑shadows/glows  
✗ Don’t change the gradient angles  
✗ Don’t recolor the wordmark beyond the approved palette

> **Assets to prepare**: Horizontal, stacked, icon‑only; light/dark; SVG/PNG/PDF; favicons; social avatars.

---

## 3) Color System

**Primary brand gradient**  
- **CGPT Purple** — `#6A5CF6`  
- **CGPT Blue** — `#5B6CFF`  
- **CGPT Pink** — `#FF59C8`

Use left→right or 45° gradient: `linear-gradient(90deg, #6A5CF6 0%, #5B6CFF 40%, #FF59C8 100%)`.

**Core colors**  
- **Indigo 600** — `#4F46E5` (buttons, links on light backgrounds)  
- **Fuchsia 500** — `#D946EF` (accents, highlights)  
- **Blue 500** — `#3B82F6` (info accents)

**Neutrals**  
- **Slate 900** — `#0F172A`  
- **Slate 700** — `#334155`  
- **Slate 500** — `#64748B`  
- **Slate 300** — `#CBD5E1`  
- **Slate 100** — `#F1F5F9`  
- **White** — `#FFFFFF`

**Functional**  
- **Success** — `#10B981`  
- **Warning** — `#F59E0B`  
- **Error** — `#EF4444`

**Accessibility**  
- Body text should be Slate 700+ on white; white on Indigo 600+ (AA at 16px).  
- Avoid small body text on Pink gradients; prefer white text overlays with soft shadow or gradient scrim.

---

## 4) Typography

**Primary typeface** — *Inter* (open‑source; modern, readable UI sans)  
**Display/Headlines** — *Poppins* (rounded geometric; matches friendly tech look)

**Web embeds (example)**  
```html
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap" rel="stylesheet">
```

**Type scale**  
- Display (Poppins 700): 56/64  
- H1 (Poppins 700): 40/48  
- H2 (Poppins 700): 32/40  
- H3 (Inter 600): 24/32  
- Body L (Inter 400): 18/28  
- Body (Inter 400): 16/26  
- Caption (Inter 500): 13/20  
- Mono (code): `ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace`

**Typesetting rules**  
- Tighten headings −2% letter‑spacing; keep body at 0%.  
- Max line‑length 60–72ch.  
- Use sentence case.

---

## 5) Buttons & Components

**Primary button**  
- Fill: **brand gradient**  
- Text: **white**  
- Radius: **12px**  
- Padding: **14px 20px**  
- Hover: shift gradient + elevate with 6px soft shadow  
- Focus: 2px focus ring in Blue 500 (WCAG‑compliant)

**Secondary button**  
- Outline: Indigo 600 2px; text Indigo 600; hover: Indigo 700 bg‑tint `rgba(79,70,229,.08)`

**Badges**  
- Info: Blue 500 on Blue 50  
- Success: Success on Green 50  
- Warning: Warning on Amber 50  
- Error: Error on Red 50

**Cards**  
- Radius **16px**, shadow `0 8px 24px rgba(2, 6, 23, .08)`, border `1px Slate 100`.

---

## 6) Iconography & Illustration

- Use simple 2px line icons; rounded corners to match Poppins.  
- Accent icons with Indigo/Fuchsia; avoid full‑color fills at small sizes.  
- For product/hero art: abstract “signal/data” motifs with subtle purple→pink gels.

---

## 7) Imagery

- Real people at work; optimistic lighting (not stocky).  
- Add gentle magenta/indigo color washes to unify with UI.  
- Avoid overly dark/edgy cyber imagery.

---

## 8) Writing Guidelines

- Lead with outcomes (accuracy, security, speed).  
- Prefer concrete proof (benchmarks, SOC‑2) over buzzwords.  
- Use active voice and verbs: *build, launch, integrate, resolve*.  
- Use numerals and specifics (e.g., “100+ integrations”, “7‑day trial”).

**Terminology**  
- Use **AI agent** (not “bot” unless customer language).  
- Use **anti‑hallucination**/**citations** when discussing accuracy.  
- Capitalization: “CustomGPT.ai”.

---

## 9) Developer Tokens (CSS)

```css
:root{
  /* Brand */
  --cgpt-purple: #6A5CF6;
  --cgpt-blue:   #5B6CFF;
  --cgpt-pink:   #FF59C8;
  --cgpt-indigo-600: #4F46E5;
  --cgpt-fuchsia-500: #D946EF;
  --cgpt-blue-500: #3B82F6;

  /* Neutrals */
  --cgpt-slate-900:#0F172A; --cgpt-slate-700:#334155; --cgpt-slate-500:#64748B;
  --cgpt-slate-300:#CBD5E1; --cgpt-slate-100:#F1F5F9; --cgpt-white:#FFFFFF;

  /* Functional */
  --cgpt-success:#10B981; --cgpt-warning:#F59E0B; --cgpt-error:#EF4444;

  /* Effects */
  --cgpt-radius-md:12px; --cgpt-radius-lg:16px;
  --cgpt-shadow-lg:0 8px 24px rgba(2,6,23,.08);
}

.btn-primary{
  background: linear-gradient(90deg,var(--cgpt-purple),var(--cgpt-blue) 40%,var(--cgpt-pink));
  color:#fff; border:none; border-radius:var(--cgpt-radius-md);
  padding:14px 20px; font-weight:600;
}
.btn-primary:hover{ filter:saturate(1.05) brightness(1.02); box-shadow:var(--cgpt-shadow-lg); }
.btn-primary:focus{ outline: 2px solid var(--cgpt-blue-500); outline-offset: 3px; }
```

**Tailwind extension (example)**  
```js
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        cgpt: {
          purple: '#6A5CF6', blue: '#5B6CFF', pink: '#FF59C8',
          indigo: {600:'#4F46E5'}, fuchsia:{500:'#D946EF'}, blue500:'#3B82F6',
          slate: {900:'#0F172A',700:'#334155',500:'#64748B',300:'#CBD5E1',100:'#F1F5F9'}
        }
      },
      borderRadius:{ md:'12px', lg:'16px' },
      boxShadow:{ cgpt:'0 8px 24px rgba(2,6,23,.08)' }
    }
  }
}
```

---

## 10) Layout & Grids

- **Container** max 1200px; gutters 24px.  
- **Grid** 12‑col, 8px baseline.  
- **Hero**: 2‑col, left text/right product art or proof badges.  
- **Cards**: 3‑col on desktop, 1‑col mobile.

---

## 11) Social & Marketing

**Avatars**: gradient brain‑chip icon; white mark for dark backgrounds.  
**Banners/Covers**: diagonal purple→pink sweep, headline in Poppins 700, subhead Inter 500.  
**Ad CTAs**: use Primary button styles; keep copy ≤ 3 words: *Start free trial*, *See demo*, *Talk to sales*.

---

## 12) Governance

- Central owner: Marketing (Brand).  
- Changes via PRD → Figma update → code tokens release.  
- Run automated contrast checks on CI for all new UI colors.

---

## 13) To‑Do (Assets Package)

1) Export all logo lockups (SVG/PNG/PDF) light/dark.  
2) Generate favicon set + app icons.  
3) Figma library with color styles & text styles.  
4) Canva templates: LinkedIn header, case‑study cover, slide master.  
5) Update website CSS tokens to match this kit.

---

*End v1*