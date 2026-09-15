"use client";

import { useEffect } from "react";
import Lenis from "lenis";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

/**
 * Inertia smooth scrolling via Lenis, driven on GSAP's ticker so the whole
 * page animates on one frame loop. Lenis lerps the NATIVE window scroll
 * (no DOM wrapping, no transformed content), so fixed elements (navbar),
 * stacking contexts (hero z-30 covering the nav), and the hero's raw
 * scrollY driver all keep working unchanged. ScrollTrigger gets updated
 * on every Lenis scroll tick so any future GSAP scrub animations stay in
 * sync with the smoothed position. Reduced-motion users: Lenis disables
 * smoothing internally via matchMedia.
 */
export function SmoothScroll() {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.6,
      smoothWheel: true,
    });

    // Canonical Lenis + ScrollTrigger integration: keep any scrubbed
    // animations reading the smoothed scroll position.
    lenis.on("scroll", () => ScrollTrigger.update());

    const raf = (time: number) => {
      lenis.raf(time * 1000);
    };
    gsap.ticker.add(raf);
    // Avoid ticker lag-jumps fighting Lenis's lerp on tab-switch.
    gsap.ticker.lagSmoothing(0);

    return () => {
      gsap.ticker.remove(raf);
      lenis.destroy();
    };
  }, []);

  return null;
}