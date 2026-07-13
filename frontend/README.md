# Frontend

This is the Next.js frontend for the Counterfeit Currency Detection System.

## Local development

```bash
npm install
npm run dev
```

Open `http://localhost:3000`.

The app talks to the FastAPI backend through `NEXT_PUBLIC_BACKEND_URL`.
If that variable is not set, it falls back to `http://127.0.0.1:8000` for local
development.

Example:

```bash
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8000 npm run dev
```

## Vercel deployment

Deploy the `frontend/` folder as the Vercel project root.

Use these settings in Vercel:

- Framework Preset: `Next.js`
- Root Directory: `frontend`
- Build Command: `npm run build`
- Output Directory: leave empty

Add this environment variable in Vercel:

```bash
NEXT_PUBLIC_BACKEND_URL=https://your-backend-url.example.com
```

Important: this project's backend is a heavy FastAPI + TensorFlow + OpenCV +
OCR service. It should usually be deployed separately from the Vercel frontend,
then connected through `NEXT_PUBLIC_BACKEND_URL`.
