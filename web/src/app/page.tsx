import Basics from "@/components/Basics";
import DeepDive from "@/components/DeepDive";
import Fairness from "@/components/Fairness";
import Fix from "@/components/Fix";
import Hero from "@/components/Hero";
import HowItWorks from "@/components/HowItWorks";
import Idea from "@/components/Idea";
import Footer from "@/components/Footer";
import { Glossary, Method } from "@/components/Method";
import Sidebar from "@/components/Sidebar";
import { Guard } from "@/components/Guard";
import PromiseCheck from "@/components/PromiseCheck";
import Results from "@/components/Results";
import SetLab from "@/components/SetLab";
import Shift from "@/components/Shift";
import WorstLaw from "@/components/WorstLaw";

export default function Home() {
  return (
    <>
      <Sidebar />
      <main id="top">
        <Hero />
        <Guard what="this chapter"><Basics /></Guard>
        <Guard what="this chapter"><Idea /></Guard>
        <Guard what="this chapter"><HowItWorks /></Guard>
        <Guard what="this chapter"><SetLab /></Guard>
        <Guard what="this chapter"><PromiseCheck /></Guard>
        <Guard what="this chapter"><Fairness /></Guard>
        <Guard what="this chapter"><WorstLaw /></Guard>
        <Guard what="this chapter"><Fix /></Guard>
        <Guard what="this chapter"><Shift /></Guard>
        <Guard what="this chapter"><Results /></Guard>
        <Guard what="this chapter"><DeepDive /></Guard>
        <Guard what="this chapter"><Method /></Guard>
        <Guard what="this chapter"><Glossary /></Guard>
      </main>
      <Footer />
    </>
  );
}
