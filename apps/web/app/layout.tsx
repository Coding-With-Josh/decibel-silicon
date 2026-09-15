import type { Metadata } from "next";
import { instrumentSans, obsydia } from "./fonts";
import { SmoothScroll } from "@/components/smooth-scroll";
import "./globals.css";

export const metadata: Metadata = {
  title: "Decibel Silicon — Low-power AI chips for all-day hearing aids",
  description:
    "Low-power AI chips for all-day hearing aids. Real-time noise reduction and speech enhancement built around a milliwatt power budget from day one — not general AI silicon scaled down.",
  openGraph: {
    title: "Decibel Silicon — Low-power AI chips for all-day hearing aids",
    description:
      "Real-time noise reduction and speech enhancement in hearing aids — chips designed around a milliwatt power budget from day one, not general AI silicon scaled down. By Joshua Idele and Bryan Zurix Whyte.",
    url: "https://decibel-silicon.vercel.app",
    siteName: "Decibel Silicon",
    locale: "en_US",
    type: "website",
    images: [
      {
        url: "https://decibel-silicon.vercel.app/chip-image.svg",
        width: 1024,
        height: 666,
        alt: "Decibel Silicon — low-power AI chips for all-day hearing aids",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "Decibel Silicon — Low-power AI chips for all-day hearing aids",
    description:
      "Low-power AI chips for all-day hearing aids — real-time noise reduction designed around a milliwatt budget.",
    images: ["https://decibel-silicon.vercel.app/chip-image.svg"],
  },
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${instrumentSans.variable} ${obsydia.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col font-sans bg-orange-500">
        <SmoothScroll />
        {children}
      </body>
    </html>
  );
}
