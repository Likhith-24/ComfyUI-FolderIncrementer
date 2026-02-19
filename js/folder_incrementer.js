import { app } from "../../scripts/app.js";

// Allow the wildcard "trigger" input to accept any link type
app.registerExtension({
    name: "Comfy.FolderIncrementer",
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        const targetNodes = [
            "FolderIncrementer",
            "FolderIncrementerReset",
            "FolderIncrementerSet",
        ];
        if (targetNodes.includes(nodeData.name)) {
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            // Allow any type to connect to the "trigger" input
            const origOnConnectInput = nodeType.prototype.onConnectInput;
            nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode, originSlot) {
                // Accept all types on the trigger slot (last optional input)
                return true;
            };
        }
    },
});
