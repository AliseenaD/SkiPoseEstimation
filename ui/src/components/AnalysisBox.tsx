import type { AnalysisResult } from "../types/analysis";

/** Props accepted by the `AnalysisBox` component. */
interface AnalysisBoxProps {
    result: AnalysisResult | null;
}

/**
 * Displays the classification result: skill level badge, confidence score with
 * a progress bar, and a bullet list of personalised coaching tips.
 *
 * Renders a placeholder prompt when `result` is null.
 */
export default function AnalysisBox({ result }: AnalysisBoxProps) {
    if (!result) {
        return (
            <div className="flex flex-col gap-6">
                <h2 className="text-primary-darker font-black text-2xl tracking-wide uppercase">
                    ANALYSIS
                </h2>
                <p className="text-secondary text-sm">
                    Upload and analyze a video to see your results here.
                </p>
            </div>
        );
    }

    const confidencePct = Math.round(result.confidence * 100);

    return (
        <div className="flex flex-col gap-6">
            <h2 className="text-primary-darker font-black text-2xl tracking-wide uppercase">
                ANALYSIS
            </h2>

            <div className="flex flex-col gap-2">
                <p className="text-secondary text-xs font-bold tracking-widest uppercase">
                    YOUR LEVEL
                </p>
                <span className="self-start bg-div-primary text-white font-black text-sm tracking-widest uppercase px-5 py-2 rounded-full">
                    {result.level}
                </span>
            </div>

            <div className="flex flex-col gap-2">
                <p className="text-secondary text-xs font-bold tracking-widest uppercase">
                    CONFIDENCE
                </p>
                <p className="text-3xl font-black text-teal-500 tracking-wide">
                    {confidencePct}%
                </p>
                <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                        className="bg-teal-500 h-2 rounded-full transition-all duration-700"
                        style={{ width: `${confidencePct}%` }}
                    />
                </div>
            </div>

            <div className="flex flex-col gap-3">
                <p className="text-secondary text-xs font-bold tracking-widest uppercase">
                    TIPS TO IMPROVE
                </p>
                <ul className="flex flex-col gap-2">
                    {result.tips.map((tip, index) => (
                        <li key={index} className="flex items-start gap-2 text-sm text-primary-darker leading-relaxed">
                            <span className="mt-1.5 w-1.5 h-1.5 rounded-full bg-div-primary shrink-0" />
                            {tip}
                        </li>
                    ))}
                </ul>
            </div>
        </div>
    );
}
