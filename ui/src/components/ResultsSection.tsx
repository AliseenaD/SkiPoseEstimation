import type { AnalysisResult } from "../types/analysis";
import PoseVideo from "./PoseVideo";
import AnalysisBox from "./AnalysisBox";

/** Props accepted by the `ResultsSection` component. */
interface ResultsSectionProps {
    result: AnalysisResult | null;
    annotatedVideoUrl: string | null;
}

/**
 * Results section shown after a successful analysis.
 *
 * Lays out `PoseVideo` and `AnalysisBox` side-by-side inside matching white
 * cards, separated by a centred "YOUR LATEST RESULTS" divider.
 */
export default function ResultsSection({ result, annotatedVideoUrl }: ResultsSectionProps) {
    return (
        <section className="bg-background py-10 px-0 md:px-20">
            <div>
                <div className="flex items-center gap-4 mb-14 px-6 md:px-0">
                    <div className="flex-1 h-px bg-secondary opacity-40" />
                    <p className="text-secondary font-black text-xs tracking-widest uppercase whitespace-nowrap">
                        YOUR LATEST RESULTS
                    </p>
                    <div className="flex-1 h-px bg-secondary opacity-40" />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-start">
                    <div className="bg-white rounded-none md:rounded-2xl shadow-md p-8">
                        <PoseVideo videoUrl={annotatedVideoUrl} />
                    </div>
                    <div className="bg-white rounded-none md:rounded-2xl shadow-md p-8">
                        <AnalysisBox result={result} />
                    </div>
                </div>
            </div>
        </section>
    );
}
