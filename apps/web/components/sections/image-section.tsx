import Image from "next/image";
import type { Ref } from "react";

/**
 * The image's own section — full viewport width/height, covers the page.
 * It lives inside the hero's sticky wrapper (see hero.tsx) and is moved up
 * over the pinned hero by ScrollTrigger when you scroll.
 *
 * Uses a plain <img> (not next/image) on purpose: /chip-image.svg is an SVG
 * wrapper around an embedded PNG, so the optimizer adds nothing here and a
 * raw img removes one more failure mode from the load path.
 */
export const ImageSection = ({
  sectionRef,
}: {
  sectionRef?: Ref<HTMLElement>;
}) => {
  return (
    <section
      ref={sectionRef}
      className="absolute inset-0 z-200 h-full w-full overflow-hidden bg-black"
      // Inline initial state: even before hydration/JS, the image sits
      // below the hero instead of covering it. GSAP overrides this
      // transform when the scrub initializes.
      // Inline initial state: even before hydration/JS, the image sits
      // below the hero instead of covering it. 110% keeps the scaled
      // image's top edge fully outside the clipped container at rest.
      style={{ transform: "translateY(110%)" }}
    >
      <Image
        width={100}
        height={100}
        src="/chip-image.svg"
        alt="Decibel Silicon chip"
        className="h-full w-full object-cover"
        loading="eager"
      />
    </section>
  );
};
