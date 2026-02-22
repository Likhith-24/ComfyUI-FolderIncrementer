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

        // ── Deep graph traversal to find a filename ──────────────────
        function findFilenameDeep(startNode, maxDepth = 12) {
            const visited = new Set();
            const queue = [{ node: startNode, depth: 0 }];

            while (queue.length > 0) {
                const { node: current, depth } = queue.shift();
                if (!current || depth > maxDepth) continue;
                if (visited.has(current.id)) continue;
                visited.add(current.id);

                const fname = getFilenameFromNode(current);
                if (fname) return fname;

                // If this looks like a Get bus node → resolve to Set nodes
                const title = (current.title || "").toLowerCase();
                const cls = (current.comfyClass || "").toLowerCase();
                const isGet = title.startsWith("get") || cls.startsWith("get")
                           || cls.includes("getnode");
                if (isGet) {
                    const setNodes = resolveGetNode(current);
                    for (const sn of setNodes) {
                        queue.push({ node: sn, depth: depth + 1 });
                        if (sn.inputs) {
                            for (const inp of sn.inputs) {
                                if (inp.link == null) continue;
                                const link = app.graph.links[inp.link];
                                if (!link) continue;
                                const upNode = app.graph.getNodeById(link.origin_id);
                                if (upNode) queue.push({ node: upNode, depth: depth + 1 });
                            }
                        }
                    }
                }

                // Follow all input links upstream
                if (current.inputs) {
                    for (const inp of current.inputs) {
                        if (inp.link == null) continue;
                        const link = app.graph.links[inp.link];
                        if (!link) continue;
                        const upNode = app.graph.getNodeById(link.origin_id);
                        if (upNode) queue.push({ node: upNode, depth: depth + 1 });
                    }
                }
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
                const result = findFilenameDeep(srcNode);
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
