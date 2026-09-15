import { Navbar } from "@/components/sections/navbar";
import { Hero } from "@/components/sections/hero";
import Image from "next/image";

export default function Home() {
  return (
    <div className="flex min-h-screen max-w-screen flex-col items-center bg-orange-600 font-sans">
      <Navbar />
      <Hero />
      <Image src="/chip-image.svg" width={100} height={100} className="size-[84rem] -mt-[10rem] mix-blend-hard-light" />
    </div>
  );
}