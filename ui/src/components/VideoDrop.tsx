import VideoIconUrl from "../assets/VideoIcon.svg?url";
import { useVideoDrop } from "../hooks/useVideoDrop";

/** Props accepted by the `VideoDrop` component. */
interface VideoDropProps {
    onAnalyze: (file: File) => void;
    isLoading: boolean;
}

/**
 * Video upload section with drag-and-drop support.
 *
 * Delegates all file and drag state to `useVideoDrop`. Shows a preview of the
 * selected video inside the drop zone, and reveals Analyze / Remove buttons
 * once a file has been accepted.
 */
export default function VideoDrop({ onAnalyze, isLoading }: VideoDropProps) {
    const {
        file,
        previewUrl,
        isDragging,
        error,
        inputRef,
        handleDragEnter,
        handleDragLeave,
        handleDragOver,
        handleDrop,
        handleFileChange,
        openFilePicker,
        clearFile,
    } = useVideoDrop();

    /** Forwards the accepted file to the parent-provided `onAnalyze` callback. */
    function handleAnalyzeClick() {
        if (file) onAnalyze(file);
    }

    return (
        <section className="bg-background py-10 px-10">
            <div className="max-w-5xl mx-auto">
                <div className="bg-white rounded-2xl shadow-md p-10">
                    <p className="text-secondary text-xs font-semibold tracking-widest uppercase mb-1">
                        UPLOAD
                    </p>
                    <h2 className="text-primary-darker font-black text-2xl tracking-wide uppercase mb-8">
                        ANALYZE MY SKIING
                    </h2>

                    <div
                        onClick={openFilePicker}
                        onDragEnter={handleDragEnter}
                        onDragLeave={handleDragLeave}
                        onDragOver={handleDragOver}
                        onDrop={handleDrop}
                        className={`
                            flex flex-col items-center justify-center gap-4
                            border-2 border-dashed rounded-xl
                            cursor-pointer select-none
                            transition-colors duration-200
                            min-h-64 p-10
                            ${isDragging
                                ? "border-div-primary bg-blue-100"
                                : "border-div-primary bg-blue-50"
                            }
                        `}
                    >
                        {previewUrl ? (
                            <video
                                src={previewUrl}
                                controls
                                className="max-h-52 rounded-lg object-contain"
                                onClick={(e) => e.stopPropagation()}
                            />
                        ) : (
                            <>
                                <div className="w-14 h-14 rounded-full bg-div-primary flex items-center justify-center">
                                    <img src={VideoIconUrl} alt="Upload video" className="w-7 h-7" />
                                </div>
                                <div className="text-center">
                                    <p className="text-primary-darker font-black text-lg tracking-wide uppercase">
                                        DRAG &amp; DROP YOUR VIDEO HERE
                                    </p>
                                    <p className="text-secondary text-sm mt-2 leading-relaxed">
                                        Drop a video of yourself on the slopes and our model will assess<br />
                                        your technique in seconds
                                    </p>
                                </div>
                            </>
                        )}
                    </div>

                    <input
                        ref={inputRef}
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={handleFileChange}
                    />

                    {error && (
                        <p className="text-red-500 text-xs mt-3">{error}</p>
                    )}

                    {file && (
                        <div className="flex items-center gap-4 mt-6">
                            <button
                                onClick={handleAnalyzeClick}
                                disabled={isLoading}
                                className="bg-div-primary hover:bg-primary-darker text-white font-black text-sm tracking-widest uppercase px-8 py-3 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                            >
                                {isLoading ? "ANALYZING..." : "ANALYZE"}
                            </button>
                            <button
                                onClick={clearFile}
                                disabled={isLoading}
                                className="text-secondary hover:text-primary-darker text-xs font-semibold tracking-widest uppercase underline transition-colors duration-200"
                            >
                                REMOVE
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </section>
    );
}
