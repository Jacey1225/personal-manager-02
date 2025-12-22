# NoManager Website

A modern web interface for the NoManager personal project management system.

## Getting Started

### Prerequisites

- Node.js 18+ installed
- npm or yarn package manager

### Installation

1. Install dependencies:
```bash
cd website
npm install
```

2. Run the development server:
```bash
npm run dev
```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Project Structure

```
website/
├── app/
│   ├── layout.tsx          # Root layout
│   ├── page.tsx            # Homepage
│   ├── projects/           # Projects page
│   └── globals.css         # Global styles
├── public/
│   └── homepage_bg.jpg     # Background image
├── package.json
├── tsconfig.json
├── tailwind.config.ts
└── next.config.mjs
```

## Features

- ✅ Beautiful homepage with background image
- ✅ Translucent navigation bar
- ✅ Projects page (placeholder)
- 🚧 Authentication (coming soon)
- 🚧 Dashboard (coming soon)
- 🚧 Widget builder (coming soon)

## Tech Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **Image Optimization**: Next.js Image component

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

## Backend Integration

The website will integrate with the FastAPI backend located in the parent `api/` directory.

API Base URL: `http://localhost:8000` (development)

## Deployment

This website is optimized for deployment on Vercel:

```bash
npm run build
```

## License

Private - All rights reserved
