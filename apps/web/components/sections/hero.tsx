"use client";

import { useEffect, useRef } from "react";
import { ImageSection } from "@/components/sections/image-section";
import { ArrowUpRight } from "lucide-react";

export const Hero = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const stickyRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const sticky = stickyRef.current;
    const image = imageRef.current;
    if (!section || !sticky || !image) return;

    // Raw scroll driver. The pinned effect is CSS position: sticky (the
    // hero stays glued while the 200vh section scrolls beneath), and the
    // image's transform is set directly from window.scrollY — no GSAP, no
    // plugin measurement to go stale.
    const apply = () => {
      const max = section.offsetHeight - window.innerHeight;
      const progress =
        max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;

      // translateY starts at 110% (extra 10% so the scaled image's top
      // edge can't peek into the viewport at rest) and ends at 0%.
      const y = 110 - 110 * progress;
      const scale = 1.15 - 0.15 * progress;
      const transform = `translateY(${y}%) scale(${scale})`;
      image.style.transform = transform;

      // Black veil over the hero: 0 at rest, 0.85 (near black) once the
      // image has fully covered. Driven by the same progress — a
      // pseudo-element reads this custom property (see .hero-dark in
      // globals.css), so no overlay element is inserted in the DOM.
      sticky.style.setProperty("--hero-dark", String(0.85 * progress));
    };

    let raf = 0;
    const onScroll = () => {
      if (raf) return;
      raf = requestAnimationFrame(() => {
        raf = 0;
        apply();
      });
    };

    apply();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);

    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (raf) cancelAnimationFrame(raf);
    };
  }, []);

  return (
    <div
      ref={sectionRef}
      className="relative w-full"
      // Inline style on purpose: the scroll range depends on this height,
      // so it must not rely on a Tailwind class that could fail to generate.
      style={{ height: "200vh" }}
    >
      {/* This sticky child is the "pin": it stays glued to the top of the
          viewport while the 200vh section scrolls beneath, then releases. */}
      <div
        ref={stickyRef}
        className="hero-dark sticky top-0 z-30 h-screen w-full overflow-hidden pointer-events-none"
      >
        <section className="flex w-full flex-col items-start justify-between px-4 pt-40">
          <h1 className="text-black font-serif text-[8rem] font-semibold">
            Decibel Silicon
          </h1>
          <div className="flex max-w-xl flex-col items-start gap-8">
            <p className="text-black text-2xl font-medium opacity-90 tracking-[-0.02em]">
              Low-power AI chips for hearing aids. Silicon, DSP, and firmware
              designed in-house for real-time audio clarity at milliwatt
              budgets.
            </p>
            <a
              href="https://youtu.be/pYhklpquIPY"
              target="_blank"
              rel="noreferrer"
              className="pointer-events-auto flex items-center justify-center px-5 py-2 bg-black text-orange-200 font-sans tracking-tight"
            >
              check out our blueprint application video
              <ArrowUpRight className="ml-2 inline-block size-3.5" />
            </a>
          </div>
        </section>
        <h1 className="font-sans tracking-[-0.06em] font-black text-[8rem] text-black/20 absolute bottom-0 right-3">COMING SOON</h1>
        <ImageSection sectionRef={imageRef} />
      </div>
    </div>
  );
};
