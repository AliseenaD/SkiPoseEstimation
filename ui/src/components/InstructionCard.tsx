/** Props accepted by the `InstructionCard` component. */
interface InstructionCardProps {
    svg: string;
    title: string;
    description: string;
}

/**
 * A single step card in the "How It Works" section.
 *
 * Displays a step icon, bold title, and descriptive text inside a white
 * card with a blue top-edge accent bar and a drop shadow.
 */
export default function InstructionCard({ svg, title, description }: InstructionCardProps) {
    return (
        <div className="bg-white rounded-none md:rounded-xl shadow-md overflow-hidden flex flex-col">
            <div className="h-1.5 bg-div-primary w-full" />
            <div className="p-7 flex flex-col gap-4">
                <img src={svg} alt={title} className="w-10 h-10 object-contain" />
                <h3 className="text-primary-darker font-black text-sm tracking-wider uppercase">
                    {title}
                </h3>
                <p className="text-secondary text-sm leading-relaxed">
                    {description}
                </p>
            </div>
        </div>
    );
}
