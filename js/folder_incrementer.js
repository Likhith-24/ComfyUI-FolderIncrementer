import { app } from "../../scripts/app.js";

/**
 * FolderIncrementer JS companion
 * - Accepts any link type on the "trigger" input
 * - Auto-extracts filename from the connected source node (Load Image,
 *   Load Video, VHS, etc.) and writes it into the source_filename widget
 * - Traverses through Set/Get bus nodes, Reroute nodes, and arbitrary
 *   graph topologies to find the original source
 * - Syncs the filename right before prompt is queued to ensure it's
 *   always up-to-date
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

        // Accept any type on trigger input
        nodeType.prototype.onConnectInput = function () { return true; };
    },

    nodeCreated(node) {
        if (node.comfyClass !== "FolderIncrementer") return;

        const FILENAME_WIDGETS = ["image", "video", "filename", "file", "audio"];

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

        // ── Input loader types that should NOT supply filenames ─────
        //    When a loader (LoadImage, LoadVideo, etc.) is directly
        //    connected to trigger, its filename is a reference/source
        //    file — not an output name.  Block these to avoid the
        //    FolderIncrementer using input filenames as output names.
        const INPUT_LOADER_TYPES = [
            "LoadImage", "Load Image", "LoadImageMask",
            "LoadVideo", "Load Video", "VHS_LoadVideo",
            "LoadAudio", "Load Audio", "VHS_LoadAudio",
            "LoadImageBatch", "LoadImagesFromDir",
        ];

        function isInputLoader(n) {
            const cls = n.comfyClass || "";
            const title = n.title || "";
            return INPUT_LOADER_TYPES.some(t => cls.includes(t) || title.includes(t));
        }

        // ── Shallow chain traversal to find a filename ───────────────
        //    Follows the direct chain (reroute / Get→Set bus) to reach
        //    the first "real" node.  Stops there — does NOT fan out
        //    across all inputs of processing nodes like VAE Decode,
        //    which previously caused wrong filenames in WAN workflows.
        //    Input loaders are blocked: their filenames are source files,
        //    not output names.
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

                // Input loader directly connected → block its filename
                if (isInputLoader(current)) {
                    return null;
                }

                // Real node — check for filename widget and stop
                return getFilenameFromNode(current);
            }
            return null;
        }

        // ── Extract filename starting from the trigger input ─────────
        function extractFilename() {
            if (!node.inputs) return null;
            for (const inp of node.inputs) {
                if (inp.name !== "trigger" || inp.link == null) continue;
                const linkInfo = app.graph.links[inp.link];
                if (!linkInfo) continue;
                const srcNode = app.graph.getNodeById(linkInfo.origin_id);
                if (!srcNode) continue;
                const result = findFilenameFromChain(srcNode);
                if (result) return result;
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
            if (connected) setTimeout(syncSourceFilename, 150);
        };

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
