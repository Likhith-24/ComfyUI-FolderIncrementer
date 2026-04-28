import { app } from "../../scripts/app.js";

/**
 * FolderIncrementer JS companion
 * - Accepts any link type on the "trigger" / "trigger_image" / "trigger_video" inputs
 * - Auto-extracts filename from the connected loader node (LoadImage,
 *   LoadVideo, VHS_LoadVideo, etc.) and writes it into the
 *   source_filename widget. The loader's filename IS what we want as
 *   our output versioning name.
 * - Traverses through Set/Get bus nodes, Reroute nodes, and arbitrary
 *   graph topologies to find the original source
 * - Honours the source_choice widget: "image" → trace trigger_image,
 *   "video" → trace trigger_video, "auto" → prefer video if connected,
 *   else image, else legacy `trigger`.
 */
app.registerExtension({
    name: "Comfy.FolderIncrementer",

    beforeRegisterNodeDef(nodeType, nodeData, app) {
        const targetNodes = [
            "FolderIncrementer",
            "FolderIncrementerReset",
            "FolderIncrementerSet",
        ];
        if (!targetNodes.includes(nodeData.name)) return;

        // Accept any type on trigger inputs
        nodeType.prototype.onConnectInput = function () { return true; };
    },

    nodeCreated(node) {
        if (node.comfyClass !== "FolderIncrementer") return;

        // Widget names commonly used by ComfyUI loaders to hold the
        // filename of the file they read.  Order matters: most-specific
        // first.  LoadImage uses "image", LoadVideo uses "video",
        // VHS_LoadVideo uses "video" too, audio loaders use "audio".
        const FILENAME_WIDGETS = ["image", "video", "filename", "file", "audio", "url"];

        // ── Check a single node for a filename widget ────────────────
        function getFilenameFromNode(n) {
            if (!n?.widgets) return null;
            for (const wName of FILENAME_WIDGETS) {
                const w = n.widgets.find(w => w.name === wName);
                if (w?.value && typeof w.value === "string" && w.value.trim()) {
                    const v = w.value.trim();
                    if (v.includes(".")) return v;
                }
            }
            return null;
        }

        // ── Resolve a Get node → find matching Set node ──────────────
        function resolveGetNode(getNode) {
            const constW = getNode.widgets?.find(
                w => w.name === "Constant" || w.name === "constant" || w.name === "value"
            );
            if (!constW?.value) return [];
            const busName = String(constW.value).trim().toLowerCase();
            if (!busName) return [];

            const results = [];
            const allNodes = app.graph._nodes || app.graph.nodes || [];
            for (const n of allNodes) {
                if (n.id === getNode.id) continue;
                const title = (n.title || "").toLowerCase();
                const cls = (n.comfyClass || "").toLowerCase();
                const isSet = title.startsWith("set") || cls.startsWith("set")
                           || cls.includes("setnode");
                if (!isSet) continue;
                const setConst = n.widgets?.find(
                    w => w.name === "Constant" || w.name === "constant" || w.name === "value"
                );
                if (setConst && String(setConst.value).trim().toLowerCase() === busName) {
                    results.push(n);
                }
            }
            return results;
        }

        // ── Node types that are transparent routing (follow through) ──
        const ROUTING_RE = /reroute|universalreroute/i;
        function isRoutingNode(n) {
            const cls = n.comfyClass || "";
            return ROUTING_RE.test(cls);
        }

        function isGetBusNode(n) {
            const cls = (n.comfyClass || "").toLowerCase();
            const title = (n.title || "").toLowerCase();
            return title.startsWith("get") || cls.startsWith("get")
                || cls.includes("getnode");
        }

        // Follow first connected input of a node (upstream one hop)
        function followFirstInput(n) {
            if (!n?.inputs) return null;
            for (const inp of n.inputs) {
                if (inp.link == null) continue;
                const link = app.graph.links[inp.link];
                if (!link) continue;
                return app.graph.getNodeById(link.origin_id) || null;
            }
            return null;
        }

        // ── Input loader detection ──────────────────────────────────
        //    Loaders are exactly the nodes whose filename we DO want.
        //    We don't block them — we extract from them.
        const INPUT_LOADER_TYPES = [
            "LoadImage", "Load Image", "LoadImageMask",
            "LoadVideo", "Load Video", "VHS_LoadVideo", "VHS_LoadVideoPath",
            "LoadAudio", "Load Audio", "VHS_LoadAudio",
            "LoadImageBatch", "LoadImagesFromDir", "LoadImagesFromDirectory",
        ];

        function isInputLoader(n) {
            const cls = n.comfyClass || "";
            const title = n.title || "";
            return INPUT_LOADER_TYPES.some(t => cls.includes(t) || title.includes(t));
        }

        // ── Shallow chain traversal to find a filename ───────────────
        //    Follows reroute / Get→Set bus to reach the first "real"
        //    node. If that node has a filename widget (e.g. LoadImage,
        //    LoadVideo), return it.  Otherwise stop — do NOT fan out
        //    across processing nodes' inputs.
        function findFilenameFromChain(startNode, maxDepth = 8) {
            let current = startNode;
            const visited = new Set();

            for (let depth = 0; depth <= maxDepth; depth++) {
                if (!current || visited.has(current.id)) return null;
                visited.add(current.id);

                // Get bus → resolve to matching Set node, then follow its input
                if (isGetBusNode(current)) {
                    const setNodes = resolveGetNode(current);
                    if (setNodes.length > 0) {
                        const upstream = followFirstInput(setNodes[0]);
                        current = upstream || setNodes[0];
                        continue;
                    }
                    return null;
                }

                // Routing/reroute → follow through transparently
                if (isRoutingNode(current)) {
                    current = followFirstInput(current);
                    continue;
                }

                // Real node (loader or otherwise) — extract filename if present
                const fn = getFilenameFromNode(current);
                if (fn) return fn;

                // No filename here. If it's a loader without one, stop.
                if (isInputLoader(current)) return null;

                // Otherwise try walking one hop upstream (helps when
                // user inserts a passthrough node between loader and
                // FolderIncrementer).
                const next = followFirstInput(current);
                if (!next) return null;
                current = next;
            }
            return null;
        }

        // ── Resolve which trigger input to traverse ──────────────────
        function findInput(name) {
            return node.inputs?.find(i => i.name === name) || null;
        }

        function getSourceNodeFromInput(inputName) {
            const inp = findInput(inputName);
            if (!inp || inp.link == null) return null;
            const linkInfo = app.graph.links[inp.link];
            if (!linkInfo) return null;
            return app.graph.getNodeById(linkInfo.origin_id) || null;
        }

        function getSourceChoice() {
            const w = node.widgets?.find(w => w.name === "source_choice");
            const v = (w?.value || "auto").toString().toLowerCase();
            return ["auto", "image", "video"].includes(v) ? v : "auto";
        }

        // ── Extract filename honouring source_choice ─────────────────
        function extractFilename() {
            if (!node.inputs) return null;

            const choice    = getSourceChoice();
            const imgSrc    = getSourceNodeFromInput("trigger_image");
            const vidSrc    = getSourceNodeFromInput("trigger_video");
            const legacySrc = getSourceNodeFromInput("trigger");

            let order;
            if (choice === "image")      order = [imgSrc, legacySrc, vidSrc];
            else if (choice === "video") order = [vidSrc, legacySrc, imgSrc];
            else                         order = [vidSrc, imgSrc, legacySrc]; // auto: video > image > legacy

            for (const src of order) {
                if (!src) continue;
                const fn = findFilenameFromChain(src);
                if (fn) return fn;
            }
            return null;
        }

        // ── Auto-fill source_filename ────────────────────────────────
        function syncSourceFilename() {
            const filename = extractFilename();
            if (!filename) return;
            const sfWidget = node.widgets?.find(w => w.name === "source_filename");
            if (sfWidget && sfWidget.value !== filename) {
                sfWidget.value = filename;
                app.graph.setDirtyCanvas(true);
            }
        }

        const origOnConnectionsChange = node.onConnectionsChange;
        node.onConnectionsChange = function (type, index, connected, link_info) {
            origOnConnectionsChange?.apply(this, arguments);
            // Re-sync whether connecting OR disconnecting: a disconnect
            // may flip auto-mode from video back to image.
            setTimeout(syncSourceFilename, 150);
        };

        // Re-sync when source_choice widget changes
        const choiceWidget = node.widgets?.find(w => w.name === "source_choice");
        if (choiceWidget) {
            const origCb = choiceWidget.callback;
            choiceWidget.callback = function (v) {
                origCb?.apply(this, arguments);
                setTimeout(syncSourceFilename, 50);
            };
        }

        const origOnExecuted = node.onExecuted;
        node.onExecuted = function (output) {
            origOnExecuted?.apply(this, arguments);
            syncSourceFilename();
        };

        // Sync before serialization (prompt queue) so Python gets fresh value
        const origOnSerialize = node.onSerialize;
        node.onSerialize = function (o) {
            syncSourceFilename();
            origOnSerialize?.apply(this, arguments);
        };

        // Initial sync + periodic retry (graph may not be fully loaded yet)
        setTimeout(syncSourceFilename, 500);
        setTimeout(syncSourceFilename, 2000);
    },
});
