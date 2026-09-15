import { ArrowUpRightFromSquare } from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import React from "react";

export const Navbar = () => {
  return (
    <div className="fixed z-20 w-screen flex items-center justify-between backdrop-blur-md py-3 px-4 md:py-4 md:px-8 bg-orange-500 0 text-white">
     <Link href="https://youtu.be/pYhklpquIPY" target="_blank" rel="noopener noreferrer">
         <button className=" flex items-center justify-center px-3 md:px-5 py-2 bg-black text-orange-200 font-sans text-sm md:text-base tracking-tight active:scale-98 hover:scale-102">
        <span className="hidden sm:inline">Blueprint Application</span>
        <span className="sm:hidden">Blueprint</span>

        <ArrowUpRightFromSquare className="ml-2 inline-block size-3.5" />
      </button>
     </Link>
      <Image src="/logo.svg" width={100} height={100} className="size-10" alt="Logo" />
      <Link href="https://github.com/Coding-With-Josh/decibel-silicon" target="_blank" rel="noopener noreferrer">
        <button className="px-3 md:px-5 py-2 bg-black text-orange-200 font-sans text-sm md:text-base tracking-tight active:scale-98 hover:scale-102">
          Source Code
        </button>
      </Link>
    </div>
  );
};
