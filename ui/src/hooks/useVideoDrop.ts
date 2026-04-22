import { useState, useCallback, useRef } from "react";

/** Shape of the object returned by `useVideoDrop`. */
interface UseVideoDrop {
    file: File | null;
    previewUrl: string | null;
    isDragging: boolean;
    error: string | null;
    inputRef: React.RefObject<HTMLInputElement | null>;
    handleDragEnter: (e: React.DragEvent) => void;
    handleDragLeave: (e: React.DragEvent) => void;
    handleDragOver: (e: React.DragEvent) => void;
    handleDrop: (e: React.DragEvent) => void;
    handleFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    openFilePicker: () => void;
    clearFile: () => void;
}

/**
 * Validates that a file is a video and does not exceed the 500 MB limit.
 * @param file - The file to validate.
 * @returns An error message string, or null if the file is acceptable.
 */
function validateVideoFile(file: File): string | null {
    if (!file.type.startsWith("video/")) {
        return "Please upload a valid video file.";
    }
    const maxBytes = 500 * 1024 * 1024;
    if (file.size > maxBytes) {
        return "File exceeds the 500 MB limit.";
    }
    return null;
}

/**
 * ViewModel hook that manages drag-and-drop video file selection.
 *
 * Handles drag events, native file-picker input, validation, and object URL
 * lifecycle (creation and revocation) for the `VideoDrop` component.
 */
export function useVideoDrop(): UseVideoDrop {
    const [file, setFile] = useState<File | null>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const inputRef = useRef<HTMLInputElement | null>(null);

    /**
     * Validates the incoming file and, if valid, replaces the current selection
     * and generates a new object URL for preview.
     * @param incoming - The candidate file from a drop or file-picker event.
     */
    function acceptFile(incoming: File) {
        const validationError = validateVideoFile(incoming);
        if (validationError) {
            setError(validationError);
            return;
        }
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setFile(incoming);
        setPreviewUrl(URL.createObjectURL(incoming));
        setError(null);
    }

    /** Sets `isDragging` to true and suppresses default browser drag behaviour. */
    const handleDragEnter = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(true);
    }, []);

    /** Sets `isDragging` to false when the cursor leaves the drop zone. */
    const handleDragLeave = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
    }, []);

    /** Prevents the browser from navigating to the file on drag-over. */
    const handleDragOver = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
    }, []);

    /** Accepts the first dropped file and clears the dragging state. */
    const handleDrop = useCallback((e: React.DragEvent) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        const dropped = e.dataTransfer.files[0];
        if (dropped) acceptFile(dropped);
    }, [previewUrl]);

    /** Accepts the first file chosen via the native file picker input. */
    const handleFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
        const selected = e.target.files?.[0];
        if (selected) acceptFile(selected);
    }, [previewUrl]);

    /** Triggers a click on the hidden file input to open the native picker. */
    const openFilePicker = useCallback(() => {
        inputRef.current?.click();
    }, []);

    /** Revokes the preview object URL and resets all file-related state. */
    function clearFile() {
        if (previewUrl) URL.revokeObjectURL(previewUrl);
        setFile(null);
        setPreviewUrl(null);
        setError(null);
        if (inputRef.current) inputRef.current.value = "";
    }

    return {
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
    };
}
