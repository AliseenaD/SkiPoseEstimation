/** The classification result returned by the analysis pipeline. */
export interface AnalysisResult {
    level: string;
    confidence: number;
    tips: string[];
}
