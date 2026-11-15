import type { Metadata } from "next";
import ExplorerPage from "@/components/explorer/Page";
import { explorer, EXPLORERS } from "@/lib/explorers";

export const dynamicParams = false;
export const generateStaticParams = () => EXPLORERS.map((e) => ({ slug: e.slug }));

export async function generateMetadata({ params }: { params: Promise<{ slug: string }> }): Promise<Metadata> {
  const e = explorer((await params).slug)!;
  return { title: `${e.short} · Conformal explorer`, description: e.blurb };
}

export default async function Page({ params }: { params: Promise<{ slug: string }> }) {
  return <ExplorerPage slug={(await params).slug} />;
}
