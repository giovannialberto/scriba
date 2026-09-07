// Scriba notification panel helper.
//
// Renders a notification-style floating card in the top-right corner of the
// main screen: dark rounded card with an icon badge, title/subtitle, a
// countdown bar along the bottom, and (in confirm mode) Ignore/Record pill
// buttons. Hovering the card pauses the countdown and reveals a ✕ dismiss
// button on the top-left corner. Unbundled CLI processes cannot post real
// Notification Center banners with action buttons, so Scriba draws its own.
//
// Usage:
//   notify-panel notify  --title T --subtitle S [--timeout 5] [--sound Glass]
//   notify-panel confirm --title T --subtitle S --yes Record --no Ignore \
//                        [--timeout 30] [--sound Glass]
//   notify-panel ... --snapshot /tmp/card.png [--hover yes|no|card]
//                        (render a PNG for design QA and exit)
//
// The subtitle is prettified: bundle identifiers are resolved to app display
// names via NSWorkspace (e.g. "company.thebrowser.browser.helper" -> "Arc").
//
// Prints exactly one line to stdout before exiting:
//   confirm mode: "yes" | "no" | "timeout"   (✕ counts as "no")
//   notify mode:  "timeout" (after the display duration, on click, or ✕)
//
// Compiled at runtime by Scriba via swiftc (see src/core/notify.rs).

import AppKit

struct Options {
    var mode = "notify"
    var title = "Scriba"
    var subtitle = ""
    var yes = "Record"
    var no = "Ignore"
    var timeout: Double = 5
    var sound: String? = nil
    var snapshot: String? = nil
    var hover: String? = nil // design QA: render an element in its hover state
}

func parseOptions() -> Options {
    var o = Options()
    var args = Array(CommandLine.arguments.dropFirst())
    if let first = args.first, !first.hasPrefix("--") {
        o.mode = first
        args.removeFirst()
    }
    var i = 0
    while i < args.count {
        let value: String? = i + 1 < args.count ? args[i + 1] : nil
        switch args[i] {
        case "--title": if let v = value { o.title = v }
        case "--subtitle", "--message": if let v = value { o.subtitle = v }
        case "--yes": if let v = value { o.yes = v }
        case "--no": if let v = value { o.no = v }
        case "--timeout": if let v = value, let t = Double(v) { o.timeout = t }
        case "--sound": o.sound = value
        case "--snapshot": o.snapshot = value
        case "--hover": o.hover = value
        default:
            i -= 1 // flag without value: don't skip the next arg
        }
        i += 2
    }
    return o
}

/// Resolve a raw process identity (bundle id or executable name) to a
/// human-friendly app name. Helper bundles resolve to their host app by
/// progressively trimming trailing segments ("company.thebrowser.browser
/// .helper" -> Arc).
func friendlyAppName(_ raw: String) -> String {
    let trimmed = raw.trimmingCharacters(in: .whitespaces)
    if !trimmed.contains(".") || trimmed.contains(" ") { return trimmed }
    var parts = trimmed.split(separator: ".").map(String.init)
    var best: String? = nil
    // Keep the shallowest resolution so helper bundles report their host app
    // ("company.thebrowser.browser.helper" resolves to "Browser Helper", but
    // "company.thebrowser.browser" resolves to "Arc" — prefer Arc).
    while parts.count >= 2 {
        let candidate = parts.joined(separator: ".")
        if let url = NSWorkspace.shared.urlForApplication(withBundleIdentifier: candidate) {
            let name = FileManager.default.displayName(atPath: url.path)
            best = name.hasSuffix(".app") ? String(name.dropLast(4)) : name
        }
        parts.removeLast()
    }
    return best ?? trimmed
}

func prettySubtitle(_ raw: String) -> String {
    let parts = raw.split(separator: ",").map {
        friendlyAppName(String($0))
    }
    var seen = Set<String>()
    let unique = parts.filter { seen.insert($0).inserted }
    return unique.joined(separator: ", ")
}

// ── Palette ──────────────────────────────────────────────────────────────────
// Brand: Scriba violet — #7c3aed (primary), #a78bfa (light variant), matching
// the website theme and the TUI's purple accents.
let brandViolet = NSColor(red: 0.486, green: 0.227, blue: 0.929, alpha: 1.0)
let brandVioletLight = NSColor(red: 0.655, green: 0.545, blue: 0.980, alpha: 1.0)

let cardColor = NSColor(red: 0.157, green: 0.153, blue: 0.145, alpha: 0.985)
let strokeColor = brandVioletLight.withAlphaComponent(0.16)
let titleColor = NSColor(white: 1.0, alpha: 0.96)
let subtitleColor = NSColor(white: 1.0, alpha: 0.55)
let badgeColor = brandViolet.withAlphaComponent(0.28)
let badgeIconColor = brandVioletLight
let primaryPillColor = brandViolet
let primaryPillHover = brandViolet.blended(withFraction: 0.18, of: .white) ?? brandViolet
let primaryPillText = NSColor(white: 1.0, alpha: 0.98)
let quietPillColor = NSColor(white: 1.0, alpha: 0.09)
let quietPillHover = NSColor(white: 1.0, alpha: 0.17)
let quietPillText = NSColor(white: 1.0, alpha: 0.85)
let closeFill = NSColor(red: 0.23, green: 0.225, blue: 0.215, alpha: 1.0)
let closeHover = NSColor(red: 0.32, green: 0.315, blue: 0.30, alpha: 1.0)
let countdownColor = brandVioletLight.withAlphaComponent(0.45)

let opts = parseOptions()
let app = NSApplication.shared
app.setActivationPolicy(.accessory)

let hasButtons = opts.mode == "confirm"
let title = opts.title
let subtitle = prettySubtitle(opts.subtitle)

// ── Result plumbing ──────────────────────────────────────────────────────────
var panelRef: NSPanel? = nil
var finished = false
func finish(_ result: String) {
    if finished { return }
    finished = true
    guard let panel = panelRef else {
        print(result)
        fflush(stdout)
        exit(0)
    }
    NSAnimationContext.runAnimationGroup(
        { ctx in
            ctx.duration = 0.15
            panel.animator().alphaValue = 0
        },
        completionHandler: {
            print(result)
            fflush(stdout)
            exit(0)
        })
}

final class Handler: NSObject {
    @objc func yes(_ sender: Any?) { finish("yes") }
    @objc func no(_ sender: Any?) { finish("no") }
    @objc func dismiss(_ sender: Any?) { finish("timeout") }
    // ✕: decline in confirm mode, plain dismissal otherwise.
    @objc func close(_ sender: Any?) { finish(hasButtons ? "no" : "timeout") }
}
let handler = Handler()

// ── Building blocks ──────────────────────────────────────────────────────────
func textWidth(_ string: String, font: NSFont) -> CGFloat {
    (string as NSString).size(withAttributes: [.font: font]).width
}

/// Button with a hover state: the fill brightens on mouse-over (with a short
/// cross-fade) and the cursor becomes a pointing hand.
class HoverButton: NSButton {
    var baseFill: NSColor = .clear
    var hoverFill: NSColor = .clear
    private var trackingArea: NSTrackingArea?

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let area = trackingArea {
            removeTrackingArea(area)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways],
            owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    override func mouseEntered(with event: NSEvent) {
        setFill(hoverFill)
        // Cursor rects only apply to key windows and this panel never becomes
        // key (non-activating), so set the cursor by hand.
        NSCursor.pointingHand.set()
    }

    override func mouseExited(with event: NSEvent) {
        setFill(baseFill)
        NSCursor.arrow.set()
    }

    func setFill(_ color: NSColor) {
        guard let layer = layer else { return }
        // The backing layer of a layer-backed view doesn't animate implicit
        // property changes; cross-fade explicitly.
        let fade = CABasicAnimation(keyPath: "backgroundColor")
        fade.fromValue = layer.backgroundColor
        fade.toValue = color.cgColor
        fade.duration = 0.15
        layer.add(fade, forKey: "fill")
        layer.backgroundColor = color.cgColor
    }
}

/// Root view: tracks hover over the whole card area to pause the countdown
/// and reveal the ✕ button.
final class HoverRoot: NSView {
    var onHoverChange: ((Bool) -> Void)?
    private var trackingArea: NSTrackingArea?

    override func updateTrackingAreas() {
        super.updateTrackingAreas()
        if let area = trackingArea {
            removeTrackingArea(area)
        }
        let area = NSTrackingArea(
            rect: bounds,
            options: [.mouseEnteredAndExited, .activeAlways],
            owner: self, userInfo: nil)
        addTrackingArea(area)
        trackingArea = area
    }

    override func mouseEntered(with event: NSEvent) { onHoverChange?(true) }
    override func mouseExited(with event: NSEvent) { onHoverChange?(false) }
}

func makePill(
    _ label: String, fill: NSColor, hoverFill: NSColor, textColor: NSColor,
    action: Selector
) -> HoverButton {
    let font = NSFont.systemFont(ofSize: 13, weight: .semibold)
    let button = HoverButton(title: label, target: handler, action: action)
    button.isBordered = false
    button.wantsLayer = true
    button.baseFill = fill
    button.hoverFill = hoverFill
    button.layer?.backgroundColor = fill.cgColor
    button.layer?.cornerRadius = 16
    button.layer?.cornerCurve = .continuous
    button.attributedTitle = NSAttributedString(
        string: label,
        attributes: [.font: font, .foregroundColor: textColor])
    let width = max(textWidth(label, font: font) + 30, 72)
    button.frame = NSRect(x: 0, y: 0, width: width, height: 32)
    return button
}

// ── Layout ───────────────────────────────────────────────────────────────────
let titleFont = NSFont.systemFont(ofSize: 15, weight: .semibold)
let subtitleFont = NSFont.systemFont(ofSize: 13, weight: .regular)

let cardHeight: CGFloat = 66
let leftPad: CGFloat = 14
let badgeSize: CGFloat = 38
let textGap: CGFloat = 12
let rightPad: CGFloat = 14
// Transparent margin on the top/left of the panel so the ✕ can straddle the
// card corner.
let closeSize: CGFloat = 24
let overhang: CGFloat = closeSize / 2

let yesPill = makePill(
    opts.yes, fill: primaryPillColor, hoverFill: primaryPillHover,
    textColor: primaryPillText, action: #selector(Handler.yes(_:)))
yesPill.keyEquivalent = "\r"
let noPill = makePill(
    opts.no, fill: quietPillColor, hoverFill: quietPillHover,
    textColor: quietPillText, action: #selector(Handler.no(_:)))

let buttonsWidth: CGFloat =
    hasButtons ? noPill.frame.width + 8 + yesPill.frame.width + 16 : 0
let naturalTextWidth = max(
    textWidth(title, font: titleFont),
    textWidth(subtitle, font: subtitleFont))
let fixedWidth = leftPad + badgeSize + textGap + rightPad + buttonsWidth
let cardWidth = min(max(fixedWidth + naturalTextWidth + 8, 320), 480)
let labelWidth = cardWidth - fixedWidth

let rootWidth = cardWidth + overhang
let rootHeight = cardHeight + overhang

let panel = NSPanel(
    contentRect: NSRect(x: 0, y: 0, width: rootWidth, height: rootHeight),
    styleMask: [.borderless, .nonactivatingPanel],
    backing: .buffered, defer: false)
panelRef = panel
panel.level = .statusBar
panel.isOpaque = false
panel.backgroundColor = .clear
panel.hasShadow = true
panel.hidesOnDeactivate = false
panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

let root = HoverRoot(
    frame: NSRect(x: 0, y: 0, width: rootWidth, height: rootHeight))
panel.contentView = root

let card = NSView(
    frame: NSRect(x: overhang, y: 0, width: cardWidth, height: cardHeight))
card.wantsLayer = true
card.layer?.backgroundColor = cardColor.cgColor
card.layer?.cornerRadius = 18
card.layer?.cornerCurve = .continuous
card.layer?.borderWidth = 1
card.layer?.borderColor = strokeColor.cgColor
root.addSubview(card)

// Icon badge: rounded square with a mic symbol.
let badge = NSView(
    frame: NSRect(
        x: leftPad, y: (cardHeight - badgeSize) / 2,
        width: badgeSize, height: badgeSize))
badge.wantsLayer = true
badge.layer?.backgroundColor = badgeColor.cgColor
badge.layer?.cornerRadius = 11
badge.layer?.cornerCurve = .continuous
card.addSubview(badge)

if let micImage = NSImage(
    systemSymbolName: "mic.fill", accessibilityDescription: nil)
{
    let config = NSImage.SymbolConfiguration(pointSize: 16, weight: .semibold)
    let imageView = NSImageView(
        frame: NSRect(x: 0, y: 0, width: badgeSize, height: badgeSize))
    imageView.image = micImage.withSymbolConfiguration(config)
    imageView.contentTintColor = badgeIconColor
    badge.addSubview(imageView)
}

// Title / subtitle stack, vertically centered.
let textX = leftPad + badgeSize + textGap
let hasSubtitle = !subtitle.isEmpty

let titleLabel = NSTextField(labelWithString: title)
titleLabel.font = titleFont
titleLabel.textColor = titleColor
titleLabel.lineBreakMode = .byTruncatingTail
let titleY: CGFloat = hasSubtitle ? cardHeight / 2 - 1 : (cardHeight - 18) / 2
titleLabel.frame = NSRect(x: textX, y: titleY, width: labelWidth, height: 18)
card.addSubview(titleLabel)

if hasSubtitle {
    let subtitleLabel = NSTextField(labelWithString: subtitle)
    subtitleLabel.font = subtitleFont
    subtitleLabel.textColor = subtitleColor
    subtitleLabel.lineBreakMode = .byTruncatingTail
    subtitleLabel.frame = NSRect(
        x: textX, y: cardHeight / 2 - 18, width: labelWidth, height: 16)
    card.addSubview(subtitleLabel)
}

if hasButtons {
    yesPill.frame.origin = NSPoint(
        x: cardWidth - rightPad - yesPill.frame.width,
        y: (cardHeight - 32) / 2)
    noPill.frame.origin = NSPoint(
        x: yesPill.frame.origin.x - 8 - noPill.frame.width,
        y: (cardHeight - 32) / 2)
    card.addSubview(noPill)
    card.addSubview(yesPill)
} else {
    // Click anywhere on a plain notification to dismiss it.
    let click = NSClickGestureRecognizer(
        target: handler, action: #selector(Handler.dismiss(_:)))
    card.addGestureRecognizer(click)
}

// Countdown bar along the card bottom, shrinking as the timeout approaches.
let barInset: CGFloat = 18
let barMaxWidth = cardWidth - barInset * 2
let countdownBar = NSView(
    frame: NSRect(x: barInset, y: 6, width: barMaxWidth, height: 3))
countdownBar.wantsLayer = true
countdownBar.layer?.backgroundColor = countdownColor.cgColor
countdownBar.layer?.cornerRadius = 1.5
card.addSubview(countdownBar)

// ✕ dismiss button straddling the top-left corner; revealed on hover.
let closeButton = HoverButton(
    title: "", target: handler, action: #selector(Handler.close(_:)))
closeButton.isBordered = false
closeButton.wantsLayer = true
closeButton.baseFill = closeFill
closeButton.hoverFill = closeHover
closeButton.layer?.backgroundColor = closeFill.cgColor
closeButton.layer?.cornerRadius = closeSize / 2
closeButton.layer?.borderWidth = 1
closeButton.layer?.borderColor = NSColor(white: 1.0, alpha: 0.18).cgColor
if let xImage = NSImage(
    systemSymbolName: "xmark", accessibilityDescription: "Dismiss")
{
    let config = NSImage.SymbolConfiguration(pointSize: 9, weight: .bold)
    closeButton.image = xImage.withSymbolConfiguration(config)
    closeButton.imagePosition = .imageOnly
    closeButton.contentTintColor = NSColor(white: 1.0, alpha: 0.9)
}
closeButton.frame = NSRect(
    x: 0, y: rootHeight - closeSize, width: closeSize, height: closeSize)
closeButton.alphaValue = 0
root.addSubview(closeButton)

// ── Countdown: ticks the bar down; hovering the card pauses it ──────────────
var remaining = opts.timeout
var hoverPaused = false

root.onHoverChange = { hovering in
    hoverPaused = hovering
    NSAnimationContext.runAnimationGroup { ctx in
        ctx.duration = 0.15
        closeButton.animator().alphaValue = hovering ? 1 : 0
    }
}

func startCountdown() {
    let tick = 0.05
    let timer = Timer(timeInterval: tick, repeats: true) { timer in
        if finished {
            timer.invalidate()
            return
        }
        if hoverPaused { return }
        remaining -= tick
        if remaining <= 0 {
            timer.invalidate()
            finish("timeout")
            return
        }
        let fraction = CGFloat(max(remaining / opts.timeout, 0))
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        countdownBar.frame.size.width = barMaxWidth * fraction
        CATransaction.commit()
    }
    RunLoop.main.add(timer, forMode: .common)
}

// ── Snapshot mode: render the card to a PNG and exit (used for design QA) ───
if let snapshotPath = opts.snapshot {
    switch opts.hover {
    case "yes": yesPill.setFill(primaryPillHover)
    case "no": noPill.setFill(quietPillHover)
    case "card":
        closeButton.alphaValue = 1
        countdownBar.frame.size.width = barMaxWidth * 0.62
    default: break
    }
    root.layoutSubtreeIfNeeded()
    if let rep = root.bitmapImageRepForCachingDisplay(in: root.bounds) {
        root.cacheDisplay(in: root.bounds, to: rep)
        if let data = rep.representation(using: .png, properties: [:]) {
            try? data.write(to: URL(fileURLWithPath: snapshotPath))
        }
    }
    print("snapshot")
    exit(0)
}

// ── Show, top-right of the main screen ───────────────────────────────────────
if let screen = NSScreen.main {
    let vf = screen.visibleFrame
    // Anchor the card (not the transparent ✕ margin) 16pt from the right edge
    // and 12pt from the top.
    panel.setFrameOrigin(
        NSPoint(
            x: vf.maxX - rootWidth - 16,
            y: vf.maxY - 12 + overhang - rootHeight))
}
panel.alphaValue = 0
panel.orderFrontRegardless()
NSAnimationContext.runAnimationGroup { ctx in
    ctx.duration = 0.18
    panel.animator().alphaValue = 1
}
if let soundName = opts.sound {
    NSSound(named: soundName)?.play()
}

startCountdown()
app.run()
