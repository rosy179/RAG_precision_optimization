import { useId } from "react";
import type { CSSProperties } from "react";

/**
 * AiRobotIcon — animated rainbow-glow robot face
 * A colorful blurred aura sphere with a glossy bean-shaped visor,
 * blinking pill eyes and a soft mirrored reflection underneath.
 * Rendered as pure SVG so the visor keeps its exact bean curve
 * and every gradient/blur scales crisply at any size.
 */

interface Props {
  size?: number;   // outer container px (default 140)
  mini?: boolean;  // compact 36px version for icon rail / avatars
}

/* Bean-shaped visor: round end lobes, subtle dip at top centre
   and a soft waist at bottom centre (viewBox 200×230). */
const VISOR_PATH =
  "M 42 82 " +
  "C 42 66 54 55 69 55 C 80 55 88 60 100 60 C 112 60 120 55 131 55 " +
  "C 146 55 158 66 158 82 " +
  "C 158 98 146 109 131 109 C 120 109 112 105 100 105 C 88 105 80 109 69 109 " +
  "C 54 109 42 98 42 82 Z";

/* Upper part of the bean, used for the glass sheen. */
const GLOSS_PATH =
  "M 42 82 " +
  "C 42 66 54 55 69 55 C 80 55 88 60 100 60 C 112 60 120 55 131 55 " +
  "C 146 55 158 66 158 82 " +
  "C 158 88 148 93 131 93 C 114 93 86 93 69 93 C 52 93 42 88 42 82 Z";

/* SVG transforms need an explicit box/origin to animate around. */
const fillBox: CSSProperties = {
  transformBox: "fill-box",
  transformOrigin: "center",
};

export default function AiRobotIcon({ size = 140, mini = false }: Props) {
  const uid = useId().replace(/:/g, "");
  const id = (name: string) => `${uid}-${name}`;
  const url = (name: string) => `url(#${id(name)})`;

  return (
    <div
      style={{
        width: mini ? 36 : size,
        height: mini ? 36 : size * 1.15,
        position: "relative",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        animation: mini ? undefined : "robot-float 5s ease-in-out infinite",
      }}
    >
      <svg
        width="100%"
        height="100%"
        viewBox={mini ? "20 15 160 160" : "0 0 200 230"}
        style={{ overflow: "visible", display: "block" }}
        role="img"
        aria-label="AI assistant"
      >
        <defs>
          <filter id={id("aura-blur")} x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="13" />
          </filter>
          <filter id={id("refl-blur")} x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="10" />
          </filter>
          <filter id={id("eye-glow")} x="-120%" y="-80%" width="340%" height="260%">
            <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor="#FFFFFF" floodOpacity="0.85" />
          </filter>

          {/* Visor body — glossy indigo, darker toward the bottom */}
          <linearGradient id={id("visor")} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#554B99" />
            <stop offset="35%" stopColor="#302770" />
            <stop offset="70%" stopColor="#1D154C" />
            <stop offset="100%" stopColor="#100A33" />
          </linearGradient>
          {/* Rim light: bright white top edge, soft lavender bottom edge */}
          <linearGradient id={id("rim")} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.95" />
            <stop offset="45%" stopColor="#FFFFFF" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#D9DCFF" stopOpacity="0.7" />
          </linearGradient>
          {/* Glass sheen fading down from the top edge */}
          <linearGradient id={id("gloss")} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.85" />
            <stop offset="55%" stopColor="#FFFFFF" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
          </linearGradient>
          {/* Aura light bouncing into the visor's lower half */}
          <linearGradient id={id("bounce")} x1="0" y1="1" x2="0" y2="0">
            <stop offset="0%" stopColor="#C8CCFF" stopOpacity="0.65" />
            <stop offset="40%" stopColor="#C8CCFF" stopOpacity="0" />
          </linearGradient>
          {/* Reflection fades out with distance from the "glass floor" */}
          <linearGradient id={id("refl-fade")} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.85" />
            <stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
          </linearGradient>
          <mask id={id("refl-mask")}>
            <rect x="0" y="158" width="200" height="60" fill={url("refl-fade")} />
          </mask>
          {/* Contains the aura colors inside one circle with a softly blurred rim */}
          <radialGradient
            id={id("aura-edge")}
            gradientUnits="userSpaceOnUse"
            cx="100" cy="95" r="70"
          >
            <stop offset="0%" stopColor="#FFFFFF" />
            <stop offset="72%" stopColor="#FFFFFF" />
            <stop offset="88%" stopColor="#FFFFFF" stopOpacity="0.55" />
            <stop offset="100%" stopColor="#FFFFFF" stopOpacity="0" />
          </radialGradient>
          <mask id={id("aura-mask")}>
            <circle cx="100" cy="95" r="70" fill={url("aura-edge")} />
          </mask>
          <clipPath id={id("visor-clip")}>
            <path d={VISOR_PATH} />
          </clipPath>
        </defs>


        {/* ── Aura sphere: pink top, orange left, cyan bottom-left, blue-violet bottom-right ── */}
        <g style={{ ...fillBox, animation: "aura-sway 9s ease-in-out infinite" }}>
          <g style={{ ...fillBox, animation: "aura-breathe 5s ease-in-out infinite" }}>
            <g mask={url("aura-mask")}>
              <g filter={url("aura-blur")}>
                <circle cx="100" cy="98" r="66" fill="#B7A6F7" opacity="0.92" />
                <circle cx="104" cy="44" r="38" fill="#FF5D7E" opacity="0.95" />
                <circle cx="152" cy="76" r="30" fill="#F857C9" opacity="0.9" />
                <circle cx="50" cy="58" r="28" fill="#FFAC63" opacity="0.9" />
                <circle cx="40" cy="90" r="22" fill="#FFD09C" opacity="0.8" />
                <circle cx="58" cy="127" r="35" fill="#3FE5DF" opacity="0.95" />
                <circle cx="94" cy="143" r="27" fill="#7FE9F2" opacity="0.85" />
                <circle cx="136" cy="128" r="36" fill="#4E76FF" opacity="0.95" />
                <circle cx="156" cy="104" r="24" fill="#8A5CFF" opacity="0.85" />
                {/* bright core so the aura glows white right around the visor */}
                <circle cx="100" cy="80" r="32" fill="#FFFFFF" opacity="0.6" />
              </g>
            </g>
          </g>
        </g>
 
        {/* ── Visor ── */}
        <g style={{ ...fillBox, animation: "robot-breathe 4s ease-in-out infinite" }}>
          <path d={VISOR_PATH} fill={url("visor")} />
          <g clipPath={url("visor-clip")}>
            {/* aura colors reflecting on the glass: cyan left, magenta right */}
            <ellipse cx="62" cy="96" rx="22" ry="16" fill="#67E8F9" opacity="0.14" />
            <ellipse cx="140" cy="94" rx="22" ry="16" fill="#FF7BD5" opacity="0.14" />
            <path d={VISOR_PATH} fill={url("bounce")} />
            <path d={GLOSS_PATH} fill={url("gloss")} />
          </g>
          <path
            d={VISOR_PATH}
            fill="none"
            stroke={url("rim")}
            strokeWidth="2.0"
            strokeLinejoin="round"
          />
 
          {/* ── Eyes: vertical white pills, blinking ── */}
          <g filter={url("eye-glow")}>
            <rect
              x="76.5" y="73.5" width="9" height="17" rx="3.5"
              fill="#FFFFFF"
              style={{ ...fillBox, animation: "eye-blink 4.4s ease-in-out infinite" }}
            />
            <rect
              x="116.5" y="73.5" width="9" height="17" rx="3.5"
              fill="#FFFFFF"
              style={{ ...fillBox, animation: "eye-blink 4.4s ease-in-out infinite" }}
            />
          </g>
        </g>

        {/* ── Reflection on the glass floor ── */}
        {!mini && (
          <g
            mask={url("refl-mask")}
            style={{ ...fillBox, animation: "reflection-pulse 5s ease-in-out infinite" }}
          >
            <g filter={url("refl-blur")}>
              <ellipse cx="68" cy="180" rx="30" ry="13" fill="#3FE5DF" opacity="0.55" />
              <ellipse cx="100" cy="184" rx="34" ry="14" fill="#7C6CFF" opacity="0.5" />
              <ellipse cx="134" cy="180" rx="28" ry="12" fill="#F857C9" opacity="0.45" />
              <ellipse cx="102" cy="176" rx="18" ry="7" fill="#FF5D7E" opacity="0.35" />
            </g>
          </g>
        )}
      </svg>
    </div>
  );
}
