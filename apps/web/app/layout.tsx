import type { Metadata } from "next";
import { instrumentSans, obsydia } from "./fonts";
import { SmoothScroll } from "@/components/smooth-scroll";
import "./globals.css";

export const metadata: Metadata = {
  title: "Decibel Silicon",
  description: "Low-power AI chips for hearing aids",
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
