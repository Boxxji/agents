import { ipcBus } from '../ipc-bus';
import type {
  AudioTransformOptions,
  ImageTransformOptions,
  RepresentationInput,
  RepresentationService,
  SummaryTransformOptions,
} from '../services/representation';

export function registerRepresentationHandlers(representationService: RepresentationService): void {
  ipcBus.handle('representation:get-available-types', async () => {
    try {
      const types = representationService.getAvailableTypes();
      return { success: true, data: types };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });

  ipcBus.handle(
    'representation:transform',
    async (_event, providerId: string, input: RepresentationInput) => {
      try {
        const result = await representationService.transform(providerId, input);
        if (!result.success) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'representation:transform-to-image',
    async (_event, input: RepresentationInput, options?: ImageTransformOptions) => {
      try {
        const result = await representationService.transformToImage(input, options);
        if (!result.success) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'representation:transform-to-summary',
    async (_event, input: RepresentationInput, options?: SummaryTransformOptions) => {
      try {
        const result = await representationService.transformToSummary(input, options);
        if (!result.success) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle(
    'representation:transform-to-audio',
    async (_event, input: RepresentationInput, options?: AudioTransformOptions) => {
      try {
        const result = await representationService.transformToAudio(input, options);
        if (!result.success) {
          return { success: false, error: result.error.message };
        }
        return { success: true, data: result.data };
      } catch (error) {
        return { success: false, error: (error as Error).message };
      }
    }
  );

  ipcBus.handle('representation:get-all-providers', async () => {
    try {
      const providers = representationService.getAllProviders().map((p) => ({
        providerId: p.providerId,
        providerName: p.providerName,
        representationType: p.representationType,
        capabilities: p.getCapabilities(),
      }));
      return { success: true, data: providers };
    } catch (error) {
      return { success: false, error: (error as Error).message };
    }
  });
}
