import { useAnalyzeVideo } from "./hooks/analyzeVideo";
import Banner from "./components/Banner";
import InstructionSection from "./components/InstructionSection";
import VideoDrop from "./components/VideoDrop";
import ResultsSection from "./components/ResultsSection";

/**
 * Root application component.
 *
 * Owns the analysis state via `useAnalyzeVideo` and distributes result data
 * down to `VideoDrop` (to trigger analysis) and `ResultsSection` (to display
 * results). `ResultsSection` is only rendered once a result is available.
 */
function App() {
    const { result, annotatedVideoUrl, isLoading, error, analyze } = useAnalyzeVideo();

    return (
        <div className="min-h-screen font-sans">
            <Banner />
            <InstructionSection />
            <VideoDrop onAnalyze={analyze} isLoading={isLoading} />
            {error && (
                <div className="bg-red-50 border border-red-200 text-red-600 text-sm px-8 py-4 text-center tracking-wide">
                    {error}
                </div>
            )}
            {result && (
                <ResultsSection result={result} annotatedVideoUrl={annotatedVideoUrl} />
            )}
        </div>
    );
}

export default App;
