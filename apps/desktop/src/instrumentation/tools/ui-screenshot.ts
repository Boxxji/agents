import { z } from 'zod';
import { getIpcBridge } from '../ipc-bridge';
import type { Tool } from '../tool-registry';

const inputSchema = z.object({
  format: z.enum(['png', 'jpeg']).optional().default('png').describe('Image format'),
  quality: z.number().min(0).max(100).optional().default(80).describe('JPEG quality (0-100)'),
});

type Input = z.infer<typeof inputSchema>;

interface Output {
  success: boolean;
  image?: {
    data: string;
    mimeType: string;
    width: number;
    height: number;
  };
  error?: string;
}

export const uiScreenshotTool: Tool<Input, Output> = {
  name: 'ui_screenshot',
  description: 'Capture a screenshot of the current app window',
  inputSchema,
  handler: async (params) => {
    try {
      const bridge = getIpcBridge();
      const result = await bridge.invoke<Output>('ui:screenshot', params, 10000);
      if (result && typeof result === 'object' && 'success' in result) {
        return result;
      }
      return { success: false, error: 'Renderer did not return screenshot data' };
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Screenshot capture failed',
      };
    }
  },
};
