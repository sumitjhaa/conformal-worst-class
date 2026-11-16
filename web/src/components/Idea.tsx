"use client";
import { CircleCheck, Layers, Percent } from "lucide-react";
import { Box, Chapter, Reveal } from "./ui";

const PROBLEMS = [
  { tone: "var(--s2)", t: "Always one answer", d: "Show a classifier a blurry photo and it still names exactly one thing, often with high confidence. It never says “I don’t know”." },
  { tone: "var(--bad)", t: "Confidence can lie", d: "A model that says “95% sure” is not right 95% of the time. Modern networks are usually over-confident." },
  { tone: "var(--violet)", t: "Mistakes are costly", d: "A doctor, a self-driving car or a content filter would rather hear “it’s A or B” than get a confident wrong answer." },
];

export default function Idea() {
  return (
    <Chapter id="idea" no={1} eyebrow="The problem" title="AI that always answers, even when it is guessing"
      lead={<>An image classifier is a program that looks at a picture and picks a label (cat, dog, truck…). It is trained on thousands of labelled examples. The trouble is <strong>how it answers</strong>.</>}>
      <ol className="numlist reveal">
        {PROBLEMS.map((p) => <li key={p.t} style={{ "--tone": p.tone } as React.CSSProperties}><h3>{p.t}</h3><p style={{ marginTop: 6 }}>{p.d}</p></li>)}
      </ol>

      <Box kind="analogy" tag="Everyday analogy">
        <p>Think of a <strong>weather forecaster</strong>. “It will rain tomorrow” is a single guess. “Tomorrow the temperature will be between
          24°C and 29°C” is a <strong>range</strong>. A good forecaster’s ranges are right about 9 times out of 10, and on hard days the range gets
          wider. I want the same from an image classifier: instead of one label, a <strong>set of labels</strong> that is usually right and grows when the picture is hard.</p>
      </Box>

      <Reveal>
        <div className="card">
          <h3>What I want, in one line</h3>
          <p style={{ font: "500 18px/1.55 var(--display)", letterSpacing: "-.01em", marginTop: 6 }}>
            For any new picture, give a set of labels such that <strong>the true label is inside the set at least 90% of the time</strong>, and make
            the sets as <strong>small</strong> as possible.
          </p>
          <dl className="defs">
            <div style={{ "--tone": "var(--good)" } as React.CSSProperties}><dt><CircleCheck />Coverage</dt><dd>How often the true label is in the set. Target: 90%.</dd></div>
            <div style={{ "--tone": "var(--s2)" } as React.CSSProperties}><dt><Layers />Set size</dt><dd>How many labels the set has. Smaller is more useful. Size 1 is a normal answer.</dd></div>
            <div style={{ "--tone": "var(--violet)" } as React.CSSProperties}><dt><Percent />α (alpha)</dt><dd>The error I allow. α = 0.1 means “wrong at most 10% of the time”.</dd></div>
          </dl>
        </div>
      </Reveal>

      <Box kind="keyidea" tag="The method has a name">
        <p>This is called <strong>conformal prediction</strong>. Its big selling point: the 90% promise is a <strong>mathematical guarantee</strong> that
          works for <em>any</em> trained model, without trusting the model’s own confidence. The next chapter shows exactly how it works, with real numbers.</p>
      </Box>
    </Chapter>
  );
}
