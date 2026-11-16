import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono, Instrument_Serif } from "next/font/google";
import { MotionRoot } from "@/components/ui";
import { THEME_INIT } from "@/lib/theme";
import "./globals.css";
import LoadToast from "@/components/LoadToast";

const sans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });
const serif = Instrument_Serif({ variable: "--font-serif", subsets: ["latin"], weight: "400", style: ["normal", "italic"] });

const description = "An interactive, plain-language tour of conformal prediction: how to make an image classifier say \"it's one of these\" with a promise you can check, and who that promise leaves out.";
export const metadata: Metadata = {
  title: "Honest Answers from AI",
  description,
  authors: [{ name: "Sumit Kumar Jha", url: "https://sumitjha.vercel.app" }],
  openGraph: { title: "Honest Answers from AI", description, type: "article" },
  twitter: { card: "summary", title: "Honest Answers from AI", description },
};
export const viewport: Viewport = {
  width: "device-width", initialScale: 1, viewportFit: "cover",
  themeColor: [{ media: "(prefers-color-scheme: light)", color: "#eff1f5" }, { media: "(prefers-color-scheme: dark)", color: "#1e1e2e" }],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable} ${serif.variable}`} suppressHydrationWarning>
      <head><script dangerouslySetInnerHTML={{ __html: THEME_INIT }} /></head>
      <body><MotionRoot>{children}<LoadToast /></MotionRoot></body>
    </html>
  );
}
