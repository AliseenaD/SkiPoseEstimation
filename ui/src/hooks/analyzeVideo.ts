import { useState } from "react";
import type { AnalysisResult } from "../types/analysis";

/** Shape of the object returned by `useAnalyzeVideo`. */
interface UseAnalyzeVideo {
    result: AnalysisResult | null;
    annotatedVideoUrl: string | null;
    isLoading: boolean;
    error: string | null;
    analyze: (file: File) => Promise<void>;
    reset: () => void;
}

/**
 * ViewModel hook that drives the video analysis pipeline.
 *
 * Posts the selected video file to the Vite dev-server middleware (`/api/analyze`),
 * which shells out to `run_classify.py` and returns the classification result along
 * with a URL for the pose-skeleton annotated video.
 */
export function useAnalyzeVideo(): UseAnalyzeVideo {
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [annotatedVideoUrl, setAnnotatedVideoUrl] = useState<string | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    /**
     * Sends the video file to `/api/analyze` and updates result state on success.
     * @param file - The ski video file selected or dropped by the user.
     */
    async function analyze(file: File): Promise<void> {
        setIsLoading(true);
        setError(null);
        setResult(null);
        setAnnotatedVideoUrl(null);

        try {
            const formData = new FormData();
            formData.append("video", file);

            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            setResult({
                level: data.level,
                confidence: data.confidence,
                tips: data.tips,
            });
            setAnnotatedVideoUrl(data.annotated_video_url);
        } catch (err) {
            setError(err instanceof Error ? err.message : "An unknown error occurred.");
        } finally {
            setIsLoading(false);
        }
    }

    /** Resets all result state back to null. */
    function reset() {
        setResult(null);
        setAnnotatedVideoUrl(null);
        setError(null);
    }

    return { result, annotatedVideoUrl, isLoading, error, analyze, reset };
}
