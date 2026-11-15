import Fairness from "@/components/Fairness";
import Fix from "@/components/Fix";
import Hero from "@/components/Hero";
import HowItWorks from "@/components/HowItWorks";
import Idea from "@/components/Idea";
import Footer from "@/components/Footer";
import { Glossary, Method } from "@/components/Method";
import Sidebar from "@/components/Sidebar";
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
        <Idea />
        <HowItWorks />
        <SetLab />
        <PromiseCheck />
        <Fairness />
        <WorstLaw />
        <Fix />
        <Shift />
        <Results />
        <Method />
        <Glossary />
      </main>
      <Footer />
    </>
  );
}
