import InstructionCard from "./InstructionCard";
import FileIconUrl from "../assets/FileIcon.svg?url";
import PlayIconUrl from "../assets/PlayIcon.svg?url";
import ReviewIconUrl from "../assets/ReviewIcon.svg?url";

/** Data shape for a single how-it-works step. */
type Instruction = {
    svg: string;
    title: string;
    description: string;
};

/** Static content for the three how-it-works steps. */
const instructions: Instruction[] = [
    {
        svg: FileIconUrl,
        title: "UPLOAD YOUR VIDEO",
        description: "Upload a short clip of yourself skiing. We recommend a 10-20 second run from the front or behind.",
    },
    {
        svg: PlayIconUrl,
        title: "POSE ANALYSIS",
        description: "The model maps your body position frame-by-frame, analyzing stance, width, edge angle, and timing.",
    },
    {
        svg: ReviewIconUrl,
        title: "GET YOUR RESULTS",
        description: "Receive your skier classification, a pose-skeleton overlay video, and personalized coaching tips to level up your technique.",
    },
];

/**
 * "How It Works" section displaying the three-step process as a card grid.
 *
 * Each step is rendered by an `InstructionCard` with an icon, title, and
 * description sourced from the static `instructions` array above.
 */
export default function InstructionSection() {
    return (
        <section className="bg-background py-10 px-0 md:px-20">
            <div>
                <h2 className="text-primary-darker font-black text-2xl tracking-widest uppercase mb-10 px-6 md:px-0">
                    HOW IT WORKS
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {instructions.map((instruction, index) => (
                        <InstructionCard
                            key={index}
                            svg={instruction.svg}
                            title={instruction.title}
                            description={instruction.description}
                        />
                    ))}
                </div>
            </div>
        </section>
    );
}
