// Scriba notification panel helper.
//
// Renders a macOS-notification-style floating panel in the top-right corner
// of the main screen: vibrancy background, rounded corners, icon, title,
// message, and (in confirm mode) Yes/No buttons. Unbundled CLI processes
// cannot post real Notification Center banners with action buttons, so Scriba
// draws its own.
//
// Usage:
//   notify-panel notify  --title T --message M [--timeout 5] [--sound Glass]
//   notify-panel confirm --title T --message M --yes Record --no Ignore \
//                        [--timeout 30] [--sound Glass]
//
// Prints exactly one line to stdout before exiting:
//   confirm mode: "yes" | "no" | "timeout"
//   notify mode:  "timeout" (after the display duration)
//
// Compiled at runtime by Scriba via swiftc (see src/core/notify.rs).

import AppKit

struct Options {
    var mode = "notify"
    var title = "Scriba"
    var message = ""
    var yes = "OK"
    var no = "Dismiss"
    var timeout: Double = 5
    var sound: String? = nil
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
        case "--message": if let v = value { o.message = v }
        case "--yes": if let v = value { o.yes = v }
        case "--no": if let v = value { o.no = v }
        case "--timeout": if let v = value, let t = Double(v) { o.timeout = t }
        case "--sound": o.sound = value
        default:
            i -= 1 // flag without value: don't skip the next arg
        }
        i += 2
    }
    return o
}

let opts = parseOptions()
let app = NSApplication.shared
app.setActivationPolicy(.accessory)

let hasButtons = opts.mode == "confirm"

// ── Layout ───────────────────────────────────────────────────────────────────
let panelWidth: CGFloat = 356
let pad: CGFloat = 14
let iconColumn: CGFloat = 44
let textWidth = panelWidth - pad * 2 - iconColumn
let titleHeight: CGFloat = 18

let messageLabel = NSTextField(wrappingLabelWithString: opts.message)
messageLabel.font = NSFont.systemFont(ofSize: 12)
messageLabel.textColor = .secondaryLabelColor
let messageHeight = min(
    messageLabel.sizeThatFits(NSSize(width: textWidth, height: 400)).height, 120)

let buttonRow: CGFloat = hasButtons ? 40 : 0
let panelHeight = pad + titleHeight + 3 + messageHeight + buttonRow + pad

let panel = NSPanel(
    contentRect: NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight),
    styleMask: [.borderless, .nonactivatingPanel],
    backing: .buffered, defer: false)
panel.level = .statusBar
panel.isOpaque = false
panel.backgroundColor = .clear
panel.hasShadow = true
panel.hidesOnDeactivate = false
panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]

let effect = NSVisualEffectView(
    frame: NSRect(x: 0, y: 0, width: panelWidth, height: panelHeight))
effect.material = .popover
effect.blendingMode = .behindWindow
effect.state = .active
effect.wantsLayer = true
effect.layer?.cornerRadius = 14
effect.layer?.masksToBounds = true
effect.layer?.borderWidth = 0.5
effect.layer?.borderColor = NSColor.separatorColor.cgColor
panel.contentView = effect

let icon = NSTextField(labelWithString: "🎙️")
icon.font = NSFont.systemFont(ofSize: 26)
icon.frame = NSRect(
    x: pad, y: panelHeight - pad - 34, width: iconColumn - 6, height: 34)
effect.addSubview(icon)

let titleLabel = NSTextField(labelWithString: opts.title)
titleLabel.font = NSFont.boldSystemFont(ofSize: 13)
titleLabel.textColor = .labelColor
titleLabel.lineBreakMode = .byTruncatingTail
titleLabel.frame = NSRect(
    x: pad + iconColumn, y: panelHeight - pad - titleHeight,
    width: textWidth, height: titleHeight)
effect.addSubview(titleLabel)

messageLabel.frame = NSRect(
    x: pad + iconColumn,
    y: panelHeight - pad - titleHeight - 3 - messageHeight,
    width: textWidth, height: messageHeight)
effect.addSubview(messageLabel)

// ── Result plumbing ──────────────────────────────────────────────────────────
var finished = false
func finish(_ result: String) {
    if finished { return }
    finished = true
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
}
let handler = Handler()

if hasButtons {
    let yesButton = NSButton(
        title: opts.yes, target: handler, action: #selector(Handler.yes(_:)))
    yesButton.bezelStyle = .rounded
    yesButton.keyEquivalent = "\r"
    yesButton.sizeToFit()
    let noButton = NSButton(
        title: opts.no, target: handler, action: #selector(Handler.no(_:)))
    noButton.bezelStyle = .rounded
    noButton.sizeToFit()
    let yesWidth = max(yesButton.frame.width, 76)
    let noWidth = max(noButton.frame.width, 76)
    yesButton.frame = NSRect(
        x: panelWidth - pad - yesWidth, y: 11, width: yesWidth, height: 24)
    noButton.frame = NSRect(
        x: panelWidth - pad - yesWidth - 8 - noWidth, y: 11,
        width: noWidth, height: 24)
    effect.addSubview(noButton)
    effect.addSubview(yesButton)
}

// ── Show, top-right of the main screen ──────────────────────────────────────
if let screen = NSScreen.main {
    let vf = screen.visibleFrame
    panel.setFrameOrigin(
        NSPoint(x: vf.maxX - panelWidth - 16, y: vf.maxY - panelHeight - 12))
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

DispatchQueue.main.asyncAfter(deadline: .now() + opts.timeout) {
    finish("timeout")
}
app.run()
