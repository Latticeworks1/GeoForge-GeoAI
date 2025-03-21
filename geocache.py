# GeoCLIP Embedding Cache Module
# Copy this single cell into your notebook to pre-cache embeddings

import torch
import torch.nn.functional as F
from geoclip import GeoCLIP

class GeoCLIPCache:
    """Pre-computes and caches location embeddings for faster GeoCLIP inference"""
    
    def __init__(self):
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model on the correct device
        self.model = GeoCLIP().to(self.device)
        
        # Move GPS gallery to the correct device
        self.gps_gallery = self.model.gps_gallery.to(self.device)
        print(f"GPS gallery loaded: {self.gps_gallery.shape} coordinates")
        
        # Pre-compute location embeddings
        print("Pre-computing location embeddings (this may take a moment)...")
        with torch.no_grad():
            self.model.eval()
            self.location_embeddings = self.model.location_encoder(self.gps_gallery)
            self.location_embeddings = F.normalize(self.location_embeddings, dim=1)
        
        print(f"✅ Location embeddings cached: {self.location_embeddings.shape}")
        print("GeoCLIP cache is ready for use!")
    
    def get_model(self):
        """Returns the pre-loaded GeoCLIP model"""
        return self.model
    
    def get_embeddings(self):
        """Returns cached location embeddings and corresponding GPS coordinates"""
        return self.location_embeddings, self.gps_gallery

# Initialize the cache (run this cell to prepare GeoCLIP for faster predictions)
geoclip_cache = GeoCLIPCache()

# You can access these variables in other cells:
# - geoclip_cache.model: The GeoCLIP model (on GPU if available)
# - geoclip_cache.location_embeddings: Pre-computed location embeddings
# - geoclip_cache.gps_gallery: GPS coordinates corresponding to embeddings
# - geoclip_cache.device: The device being used (cuda or cpu)
