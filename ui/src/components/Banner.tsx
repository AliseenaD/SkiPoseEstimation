import bannerUrl from "../assets/banner.webp";

/**
 * Full-width hero banner displayed at the top of the page.
 *
 * Renders the ski mountain photo with a vignette overlay and the app's
 * headline "FIND YOUR LEVEL" vertically centred over the image.
 */
export default function Banner() {
    return (
        <div className="relative w-full h-130 overflow-hidden">
            <img
                src={bannerUrl}
                alt="Ski mountain"
                className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-vignette" />
            <div className="absolute inset-0 flex items-center left-12">
                <h1 className="font-black text-6xl tracking-widest leading-tight uppercase">
                    <span className="text-white">FIND YOUR</span><br />
                    <span className="text-primary-darker">LEVEL</span>
                </h1>
            </div>
        </div>
    );
}
