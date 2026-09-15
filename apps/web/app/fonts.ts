import localFont from "next/font/local";
import { Instrument_Sans } from "next/font/google";

export const instrumentSans = Instrument_Sans({
  variable: "--font-instrument-sans",
  display: "swap",
  subsets: ["latin"],
});

export const tbjNeutral = localFont({
  src: [
    { path: "../fonts/TBJNeuetraDemo-Thin-BF6a28d7acae200.ttf", weight: "100", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-ExtraLight-BF6a28d7ac86967.ttf", weight: "200", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-Light-BF6a28d7ac97b3b.ttf", weight: "300", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-Regular-BF6a28d7ac6c666.ttf", weight: "400", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-Medium-BF6a28d7ac6ea90.ttf", weight: "500", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-SemiBold-BF6a28d7ac84c37.ttf", weight: "600", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-Bold-BF6a28d7aca2ee3.ttf", weight: "700", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-ExtraBold-BF6a28d7ace34fb.ttf", weight: "800", style: "normal" },
    { path: "../fonts/TBJNeuetraDemo-Black-BF6a28d7ac6c66a.ttf", weight: "900", style: "normal" },
  ],
  variable: "--font-tbj-neutral",
  display: "swap",
  preload: false,
});

export const obsydia = localFont({
  src: [
    { path: "../fonts/ObsydiaDEMO-Regular-BF6a637423a188d.ttf", weight: "400", style: "normal" },
  ],
  variable: "--font-obsydia",
  display: "swap",
  preload: false,
});
