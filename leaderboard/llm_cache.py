#!/usr/bin/env python3
"""
llm_cache.py - Enhanced caching system for LLM responses and prompts

Stores both prompts and responses with metadata for different LLM providers.
Supports cache invalidation, statistics, and export functionality.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import pandas as pd
from datetime import datetime

from .llm_prompts import LLMProvider, EvaluationType
from .llm_clients import LLMResponse

@dataclass
class CacheEntry:
    """Complete cache entry with prompt, response, and metadata"""
    cache_key: str
    prompt_info: Dict[str, Any]
    response: Dict[str, Any]  # LLMResponse as dict
    timestamp: str
    cache_version: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CacheEntry':
        """Create from dictionary"""
        return cls(**data)

class LLMCacheManager:
    """Enhanced cache manager for LLM responses"""
    
    def __init__(self, cache_dir: Path = Path("llm_cache"), cache_version: str = "2.0"):
        self.cache_dir = cache_dir
        self.cache_version = cache_version
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for organization
        (self.cache_dir / "detailed").mkdir(exist_ok=True)
        (self.cache_dir / "multiview").mkdir(exist_ok=True)
        (self.cache_dir / "pairwise").mkdir(exist_ok=True)
        (self.cache_dir / "exports").mkdir(exist_ok=True)
    
    def generate_cache_key(
        self,
        evaluation_type: EvaluationType,
        session_id: str,
        provider: LLMProvider,
        trial: int = 0,
        **kwargs
    ) -> str:
        """Generate cache key for LLM request"""
        
        # Base key components
        key_parts = [
            evaluation_type.value,
            provider.value,
            session_id,
            str(trial)
        ]
        
        # Add evaluation-specific components
        if evaluation_type == EvaluationType.DETAILED_SCORING:
            mesh_count = kwargs.get("mesh_count", 0)
            mesh_index = kwargs.get("mesh_index", 0)
            key_parts.extend([str(mesh_count), str(mesh_index)])
            
        elif evaluation_type == EvaluationType.MULTIVIEW_SCORING:
            mesh_count = kwargs.get("mesh_count", 0)
            mesh_index = kwargs.get("mesh_index", 0)
            views = kwargs.get("views", [])
            views_str = "_".join(sorted(views))
            key_parts.extend([str(mesh_count), str(mesh_index), views_str])
            
        elif evaluation_type == EvaluationType.PAIRWISE_COMPARISON:
            concept_session = kwargs.get("concept_session_id", "")
            model_a = kwargs.get("model_a", "")
            model_b = kwargs.get("model_b", "")
            key_parts.extend([concept_session, model_a, model_b])
        
        # Create hash for consistent key length
        key_data = "_".join(key_parts)
        cache_hash = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"{evaluation_type.value}_{provider.value}_{cache_hash}"
    
    def save_response(
        self,
        cache_key: str,
        llm_response: LLMResponse,
        additional_metadata: Dict[str, Any] = None
    ) -> None:
        """Save LLM response to cache"""
        
        # Create cache entry
        cache_entry = CacheEntry(
            cache_key=cache_key,
            prompt_info=llm_response.prompt_info,
            response=llm_response.to_dict(),
            timestamp=datetime.now().isoformat(),
            cache_version=self.cache_version
        )
        
        # Add additional metadata if provided
        if additional_metadata:
            cache_entry.prompt_info.update(additional_metadata)
        
        # Determine subdirectory based on evaluation type
        eval_type = llm_response.prompt_info.get("evaluation_type", "unknown")
        if "multiview" in eval_type:
            subdir = "multiview"
        elif "pairwise" in eval_type:
            subdir = "pairwise"
        else:
            subdir = "detailed"
        
        # Save to file
        cache_file = self.cache_dir / subdir / f"{cache_key}.json"
        
        try:
            with cache_file.open("w") as f:
                json.dump(cache_entry.to_dict(), f, indent=2)
            print(f"   💾 Cached response: {cache_key}")
        except IOError as e:
            print(f"   ⚠️  Failed to save cache {cache_key}: {e}")
    
    def load_response(self, cache_key: str) -> Optional[CacheEntry]:
        """Load cached response"""
        
        # Try to find cache file in subdirectories
        search_dirs = ["detailed", "multiview", "pairwise"]
        
        for subdir in search_dirs:
            cache_file = self.cache_dir / subdir / f"{cache_key}.json"
            if cache_file.exists():
                try:
                    with cache_file.open("r") as f:
                        data = json.load(f)
                    
                    # Check cache version compatibility
                    cache_version = data.get("cache_version", "1.0")
                    if cache_version != self.cache_version:
                        print(f"   ⚠️  Cache version mismatch for {cache_key}: {cache_version} vs {self.cache_version}")
                        return None
                    
                    print(f"   📋 Using cached response: {cache_key}")
                    return CacheEntry.from_dict(data)
                    
                except (json.JSONDecodeError, IOError) as e:
                    print(f"   ⚠️  Failed to load cache {cache_key}: {e}")
                    return None
        
        return None
    
    def has_cached_response(self, cache_key: str) -> bool:
        """Check if response is cached"""
        return self.load_response(cache_key) is not None
    
    def clear_cache(self, pattern: str = None, evaluation_type: EvaluationType = None) -> int:
        """Clear cached responses"""
        import glob
        
        cleared_count = 0
        
        if evaluation_type:
            # Clear specific evaluation type
            if evaluation_type == EvaluationType.MULTIVIEW_SCORING:
                search_dir = self.cache_dir / "multiview"
            elif evaluation_type == EvaluationType.PAIRWISE_COMPARISON:
                search_dir = self.cache_dir / "pairwise"
            else:
                search_dir = self.cache_dir / "detailed"
            
            files = list(search_dir.glob("*.json"))
            
        elif pattern:
            # Clear by pattern across all subdirectories
            files = []
            for subdir in ["detailed", "multiview", "pairwise"]:
                files.extend((self.cache_dir / subdir).glob(f"*{pattern}*.json"))
        else:
            # Clear all cache files
            files = []
            for subdir in ["detailed", "multiview", "pairwise"]:
                files.extend((self.cache_dir / subdir).glob("*.json"))
        
        for file_path in files:
            try:
                file_path.unlink()
                cleared_count += 1
            except Exception as e:
                print(f"   ⚠️  Failed to delete {file_path}: {e}")
        
        return cleared_count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            "total_files": 0,
            "detailed_responses": 0,
            "multiview_responses": 0,
            "pairwise_responses": 0,
            "total_size_mb": 0,
            "providers": {},
            "evaluation_types": {},
            "oldest_entry": None,
            "newest_entry": None,
        }
        
        all_files = []
        for subdir in ["detailed", "multiview", "pairwise"]:
            subdir_path = self.cache_dir / subdir
            subdir_files = list(subdir_path.glob("*.json"))
            all_files.extend(subdir_files)
            
            if subdir == "detailed":
                stats["detailed_responses"] = len(subdir_files)
            elif subdir == "multiview":
                stats["multiview_responses"] = len(subdir_files)
            elif subdir == "pairwise":
                stats["pairwise_responses"] = len(subdir_files)
        
        stats["total_files"] = len(all_files)
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in all_files)
        stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        
        # Analyze cache entries for detailed stats
        timestamps = []
        for cache_file in all_files:
            try:
                with cache_file.open("r") as f:
                    data = json.load(f)
                
                # Count by provider
                provider = data.get("prompt_info", {}).get("provider", "unknown")
                stats["providers"][provider] = stats["providers"].get(provider, 0) + 1
                
                # Count by evaluation type
                eval_type = data.get("prompt_info", {}).get("evaluation_type", "unknown")
                stats["evaluation_types"][eval_type] = stats["evaluation_types"].get(eval_type, 0) + 1
                
                # Track timestamps
                timestamp = data.get("timestamp")
                if timestamp:
                    timestamps.append(timestamp)
                    
            except Exception:
                continue  # Skip corrupted files
        
        # Find oldest and newest entries
        if timestamps:
            timestamps.sort()
            stats["oldest_entry"] = timestamps[0]
            stats["newest_entry"] = timestamps[-1]
        
        return stats
    
    def export_cache_data(self, output_file: Path = None) -> Path:
        """Export cache data to CSV for analysis"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.cache_dir / "exports" / f"cache_export_{timestamp}.csv"
        
        # Collect all cache entries
        cache_data = []
        
        for subdir in ["detailed", "multiview", "pairwise"]:
            subdir_path = self.cache_dir / subdir
            for cache_file in subdir_path.glob("*.json"):
                try:
                    with cache_file.open("r") as f:
                        data = json.load(f)
                    
                    # Extract key information
                    prompt_info = data.get("prompt_info", {})
                    response_data = data.get("response", {})
                    
                    entry = {
                        "cache_key": data.get("cache_key", ""),
                        "timestamp": data.get("timestamp", ""),
                        "provider": prompt_info.get("provider", ""),
                        "model": prompt_info.get("model", ""),
                        "evaluation_type": prompt_info.get("evaluation_type", ""),
                        "session_id": response_data.get("metadata", {}).get("session_id", ""),
                        "success": response_data.get("success", False),
                        "error": response_data.get("error", ""),
                        "image_count": prompt_info.get("image_count", 0),
                        "has_parsed_json": response_data.get("parsed_json") is not None,
                        "content_length": len(response_data.get("content", "")),
                    }
                    
                    cache_data.append(entry)
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to process {cache_file}: {e}")
                    continue
        
        # Create DataFrame and save to CSV
        df = pd.DataFrame(cache_data)
        df.to_csv(output_file, index=False)
        
        print(f"📊 Cache data exported to: {output_file}")
        print(f"   Total entries: {len(cache_data)}")
        
        return output_file
    
    def validate_cache_integrity(self) -> Dict[str, Any]:
        """Validate cache integrity and identify issues"""
        issues = {
            "corrupted_files": [],
            "version_mismatches": [],
            "missing_fields": [],
            "invalid_json_responses": [],
            "total_checked": 0,
            "total_issues": 0,
        }
        
        for subdir in ["detailed", "multiview", "pairwise"]:
            subdir_path = self.cache_dir / subdir
            for cache_file in subdir_path.glob("*.json"):
                issues["total_checked"] += 1
                
                try:
                    with cache_file.open("r") as f:
                        data = json.load(f)
                    
                    # Check required fields
                    required_fields = ["cache_key", "prompt_info", "response", "timestamp"]
                    for field in required_fields:
                        if field not in data:
                            issues["missing_fields"].append(str(cache_file))
                            break
                    
                    # Check cache version
                    cache_version = data.get("cache_version", "1.0")
                    if cache_version != self.cache_version:
                        issues["version_mismatches"].append(str(cache_file))
                    
                    # Check response JSON validity
                    response = data.get("response", {})
                    if response.get("success") and not response.get("parsed_json"):
                        issues["invalid_json_responses"].append(str(cache_file))
                    
                except json.JSONDecodeError:
                    issues["corrupted_files"].append(str(cache_file))
                except Exception as e:
                    issues["corrupted_files"].append(f"{cache_file}: {e}")
        
        issues["total_issues"] = sum(len(v) for k, v in issues.items() if isinstance(v, list))
        
        return issues

# -----------------------------------------------------------------------------
# Global cache instance
# -----------------------------------------------------------------------------

# Create global cache manager instance
_cache_manager = None

def get_cache_manager() -> LLMCacheManager:
    """Get global cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = LLMCacheManager()
    return _cache_manager

def set_cache_directory(cache_dir: Path) -> None:
    """Set cache directory and reinitialize cache manager"""
    global _cache_manager
    _cache_manager = LLMCacheManager(cache_dir)

# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------

def save_llm_response(cache_key: str, response: LLMResponse, metadata: Dict[str, Any] = None) -> None:
    """Save LLM response to cache"""
    cache_manager = get_cache_manager()
    cache_manager.save_response(cache_key, response, metadata)

def load_llm_response(cache_key: str) -> Optional[LLMResponse]:
    """Load LLM response from cache"""
    cache_manager = get_cache_manager()
    cache_entry = cache_manager.load_response(cache_key)
    
    if cache_entry:
        # Reconstruct LLMResponse from cached data
        response_data = cache_entry.response.copy()
        # Convert provider string back to enum
        if 'provider' in response_data and isinstance(response_data['provider'], str):
            response_data['provider'] = LLMProvider(response_data['provider'])
        return LLMResponse(**response_data)
    
    return None

def has_cached_llm_response(cache_key: str) -> bool:
    """Check if LLM response is cached"""
    cache_manager = get_cache_manager()
    return cache_manager.has_cached_response(cache_key) 