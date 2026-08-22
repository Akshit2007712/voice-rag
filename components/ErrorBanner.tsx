import { PixelWarnIcon } from "./PixelIcons";

interface ErrorBannerProps {
  message: string;
  onDismiss: () => void;
}

export function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div
      role="alert"
      className="mt-5 flex items-start gap-3 border-2 border-ink bg-brick px-4 py-3 text-cream shadow-pixel-sm"
    >
      <PixelWarnIcon className="mt-0.5 h-5 w-5 shrink-0" />
      <p className="flex-1 font-body text-sm leading-relaxed">{message}</p>
      <button
        type="button"
        onClick={onDismiss}
        aria-label="Dismiss error"
        className="shrink-0 border-2 border-cream/70 px-2 py-0.5 font-pixel text-[9px] hover:bg-cream/10"
      >
        X
      </button>
    </div>
  );
}
