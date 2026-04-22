/** Props accepted by the `PoseVideo` component. */
interface PoseVideoProps {
    videoUrl: string | null;
}

/**
 * Displays the pose-skeleton annotated video produced by the analysis pipeline.
 *
 * Shows a dark placeholder when no video URL is available yet.
 */
export default function PoseVideo({ videoUrl }: PoseVideoProps) {
    return (
        <div className="flex flex-col gap-3">
            <h2 className="text-primary-darker font-black text-2xl tracking-wide uppercase">
                POSE SKELETON OVERLAY
            </h2>
            <div className="bg-primary-darker rounded-xl overflow-hidden aspect-video flex items-center justify-center">
                {videoUrl ? (
                    <video
                        src={videoUrl}
                        controls
                        className="w-full h-full object-contain"
                    />
                ) : (
                    <p className="text-secondary text-xs tracking-widest uppercase">
                        No video yet
                    </p>
                )}
            </div>
        </div>
    );
}
