import type { ReactNode } from "react";

interface PixelPanelProps {
  label: string;
  icon?: ReactNode;
  tone?: "cream" | "alert" | "pipe";
  children: ReactNode;
  className?: string;
  right?: ReactNode;
}

const toneTabBg: Record<NonNullable<PixelPanelProps["tone"]>, string> = {
  cream: "bg-coin",
  alert: "bg-alert text-cream",
  pipe: "bg-pipe text-cream",
};

/**
 * The level's recurring "dialogue box" motif: a thick pixel-bordered
 * panel with a tab-label bitten out of its top edge, like an NPC
 * textbox or item-inventory slot.
 */
export function PixelPanel({
  label,
  icon,
  tone = "cream",
  children,
  className = "",
  right,
}: PixelPanelProps) {
  return (
    <section className={`relative mt-5 ${className}`}>
      <div
        className={`absolute -top-4 left-4 z-10 flex items-center gap-1.5 border-2 border-ink px-2 py-1 font-pixel text-[9px] uppercase tracking-wide shadow-pixel-sm sm:text-[10px] ${toneTabBg[tone]}`}
      >
        {icon}
        {label}
      </div>
      {right && <div className="absolute -top-4 right-4 z-10">{right}</div>}
      <div className="border-2 border-ink bg-cream-panel p-4 pt-6 shadow-pixel sm:p-5 sm:pt-7">
        {children}
      </div>
    </section>
  );
}
