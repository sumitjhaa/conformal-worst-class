"use client";
import dynamic from "next/dynamic";
import type { ComponentType } from "react";
import type { DynamicOptionsLoadingProps } from "next/dynamic";
import { LoadFailed } from "../Guard";
import { explorer } from "@/lib/explorers";
import { Loading } from "./kit";
import { Shell } from "./Shell";

type View = ComponentType<{ p: string }>;
const wait = ({ error }: DynamicOptionsLoadingProps) => (error ? <LoadFailed what="this explorer" error={error} /> : <Loading what="explorer" />);
// one chunk per explorer, so a page only downloads its own view (next/dynamic needs the options inline)
const VIEWS: Record<string, View> = {
  "set-lab": dynamic(() => import("./views/SetLabX"), { loading: wait, ssr: false }),
  leaderboard: dynamic(() => import("./views/Leaderboard"), { loading: wait, ssr: false }),
  inspector: dynamic(() => import("./views/Inspector"), { loading: wait, ssr: false }),
  "left-out": dynamic(() => import("./views/LeftOut"), { loading: wait, ssr: false }),
  "worst-class": dynamic(() => import("./views/WorstClass"), { loading: wait, ssr: false }),
  simulator: dynamic(() => import("./views/Simulator"), { loading: wait, ssr: false }),
  grouping: dynamic(() => import("./views/Grouping"), { loading: wait, ssr: false }),
  shift: dynamic(() => import("./views/ShiftLab"), { loading: wait, ssr: false }),
  aci: dynamic(() => import("./views/Aci"), { loading: wait, ssr: false }),
  ood: dynamic(() => import("./views/Ood"), { loading: wait, ssr: false }),
  calibration: dynamic(() => import("./views/Calibration"), { loading: wait, ssr: false }),
  tuning: dynamic(() => import("./views/Tuning"), { loading: wait, ssr: false }),
  confusion: dynamic(() => import("./views/Confusion"), { loading: wait, ssr: false }),
  diff: dynamic(() => import("./views/Diff"), { loading: wait, ssr: false }),
  features: dynamic(() => import("./views/Features"), { loading: wait, ssr: false }),
  imagenet: dynamic(() => import("./views/ImageNet"), { loading: wait, ssr: false }),
  draw: dynamic(() => import("./views/Draw"), { loading: wait, ssr: false }),
};

export default function ExplorerPage({ slug }: { slug: string }) {
  return <Shell info={explorer(slug)!} view={VIEWS[slug]} />;
}
