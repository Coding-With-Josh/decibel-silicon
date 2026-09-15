export const Hero = () => {
  return (
    <section className="flex w-full flex-col items-start justify-between px-4 pt-40">
      <h1 className="text-black font-serif text-[8rem] font-semibold">
        Decibel Silicon
      </h1>
      <div className="flex flex-col items-start gap-8 max-w-xl">
        <p className="text-black text-2xl font-medium opacity-90 tracking-[-0.02em]">
          Low-power AI chips for hearing aids. Silicon, DSP, and firmware
          designed in-house for real-time audio clarity at milliwatt budgets.
        </p>
        <button className="px-5 py-2 bg-black text-orange-200 font-sans tracking-tight">
          Join the waitlist
        </button>
      </div>
    </section>
  );
};
