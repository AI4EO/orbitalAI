import os
import random
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import shapely.geometry
import shapely.ops
from cv2 import BORDER_CONSTANT, INTER_LINEAR, warpAffine
from eolearn.core import EOPatch, EOTask, FeatureType
from eolearn.core.utils.parsing import FeaturesSpecification
from eolearn.io import ExportToTiffTask
from eolearn.mask.utils import resize_images
from s22hsi_constants import (
    DIRECTIONS,
    HSI_BANDS,
    HSI_CW,
    HSI_IRR,
    RAND_MEAN,
    RAND_STD,
    RELATIVE_SHIFT,
    S2_BANDS,
    S2_CW,
    ProcessingLevels,
)
from scipy.interpolate import CubicSpline
from sentinelhub import (
    BBox,
    DataCollection,
    SentinelHubCatalog,
    SHConfig,
    parse_time,
    pixel_to_utm,
)
from sentinelhub.aws.request import AwsProductRequest
from tqdm.auto import tqdm


class CalculateRadianceTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
    ):
        """Calculate radiances from reflectances using solar irradiance and distance Earth-Sun.

        :param input_feature: Input feature holding reflectance values.
        :param output_feature: Output feature with radiance values.
        """
        self.input_feature = self.parse_feature(input_feature)
        self.output_feature = self.parse_feature(output_feature)

    def execute(self, eopatch):
        assert all(
            [
                isinstance(eopatch.scalar[f"sol_irr_{band}"], np.ndarray)
                for band in S2_BANDS
            ]
        )
        assert isinstance(eopatch.scalar["earth_sun_dist"], np.ndarray)

        factor = (
            np.cos(np.radians(eopatch.data["sunZenithAngles"]))
            * eopatch.scalar["earth_sun_dist"][:, np.newaxis, np.newaxis, :]
            / np.pi
        )
        solar_irradiances = np.concatenate(
            [eopatch.scalar[f"sol_irr_{band}"][:, :, np.newaxis] for band in S2_BANDS],
            axis=-1,
        )

        radiances = (
            eopatch[self.input_feature]
            * factor
            * solar_irradiances[:, :, np.newaxis, :]
        )
        eopatch[self.output_feature] = radiances
        return eopatch


class CalculateReflectanceTask(EOTask):
    def __init__(
        self,
        feature: FeaturesSpecification,
        processing_level: ProcessingLevels,
    ):
        """Calculate reflectances from radiances using solar irradiance and distance
        Earth-Sun if L1C level is requested. Otherwise radiance are returned.

        :param feature: Input feature holding radiance values and name of new feature.
        :param processing_level: Processing level desired. If L1A or L1B no conversion to reflectances is applied.
        """
        self.feature = self.parse_renamed_feature(feature)
        self.processing_level = processing_level

    def execute(self, eopatch: EOPatch) -> EOPatch:
        if self.processing_level.value == ProcessingLevels.L1C.value:
            assert (
                len(eopatch.timestamp) == 1
            ), "Implementation supports one timestamp only"
            assert isinstance(eopatch.scalar["earth_sun_dist"], np.ndarray)

            factor = (
                np.cos(np.radians(eopatch.data["sunZenithAngles"]))
                * eopatch.scalar["earth_sun_dist"][:, np.newaxis, np.newaxis, :]
                / np.pi
            )

            reflectances = eopatch[self.feature[0]][self.feature[1]] / (
                factor * HSI_IRR.reshape(1, 1, 1, len(HSI_IRR))
            )
            eopatch[self.feature[0]][self.feature[2]] = reflectances
        else:
            eopatch[self.feature[0]][self.feature[2]] = eopatch[self.feature[0]][
                self.feature[1]
            ]
        return eopatch


class AddMetadataTask(EOTask):
    def __init__(self, config: Optional[SHConfig] = None):
        """Download Sentinel-2 metadata necessary to compute radiances

        :param config: Optional Sentinel Hub configuration file.
        """
        self.config = config

    @staticmethod
    def filter_and_sort_tiles(
        tiles: List[Dict], timestamps: List[datetime]
    ) -> List[Dict]:
        filtered_tiles = []
        available_dates = []
        for tile in tiles:
            tile_dt = parse_time(tile["properties"]["datetime"], ignoretz=True)
            if (tile_dt in timestamps) and not (tile_dt.date() in available_dates):
                tile["timestamp"] = tile_dt
                available_dates.append(tile_dt.date())
                filtered_tiles.append(tile)
        if len(filtered_tiles) != len(timestamps):
            raise ValueError(
                f"Expected {len(timestamps)} tiles, got {len(filtered_tiles)}!"
            )
        return sorted(filtered_tiles, key=lambda item: item["timestamp"])

    def execute(self, eopatch: EOPatch, **kwargs) -> EOPatch:
        if not all([eopatch, eopatch.bbox, eopatch.timestamp]):
            raise ValueError(
                "AddMetadataTask needs eopatch to have bbox and temporal data!"
            )

        # the metadata info location has been changed since 2022-01-25
        catalog = SentinelHubCatalog(self.config)
        query = catalog.search(
            collection=DataCollection.SENTINEL2_L2A,
            bbox=eopatch.bbox,
            time=[eopatch.timestamp[0], eopatch.timestamp[-1]],
        )
        tiles = self.filter_and_sort_tiles(list(query), eopatch.timestamp)

        dim = len(eopatch.timestamp)
        eopatch.scalar["earth_sun_dist"] = np.ones((dim, 1))
        for s2_band in S2_BANDS:
            eopatch.scalar[f"sol_irr_{s2_band}"] = np.ones((dim, 1))

        for tile_idx, (tile, timestamp) in enumerate(zip(tiles, eopatch.timestamp)):
            metadata = AwsProductRequest(
                product_id=tile["id"], bands=[], metafiles=["metadata"]
            ).get_data()

            index_refl_conversion = 4
            if timestamp >= datetime.strptime("2022-01-25", "%Y-%m-%d"):
                index_refl_conversion = 5

            assert (
                metadata[0][0][1][index_refl_conversion][0].tag == "U"
            ), "Issue indexing metadata"
            assert (
                metadata[0][0][1][index_refl_conversion][1].tag
                == "Solar_Irradiance_List"
            ), "Issue indexing metadata"

            eopatch.scalar["earth_sun_dist"][tile_idx] = float(
                metadata[0][0][1][index_refl_conversion][0].text
            )

            solar_irradiance_list = metadata[0][0][1][index_refl_conversion][1]
            for s2_idx, s2_band in enumerate(DataCollection.SENTINEL2_L1C.bands):
                if s2_band.name in S2_BANDS:
                    eopatch.scalar[f"sol_irr_{s2_band.name}"][tile_idx] = float(
                        solar_irradiance_list[s2_idx].text
                    )

        return eopatch


class SpectralResamplingTask(EOTask):
    def __init__(
        self,
        feature_bands: FeaturesSpecification,
    ):
        """Apply spline interpolation to Sentinel-2 bands to HSI central wavelength.

        :param feature_bands: Bands in EOPatch to be interpolated.
        """
        self.feature = self.parse_renamed_feature(feature_bands)

    def execute(self, eopatch: EOPatch) -> EOPatch:
        bands = eopatch[self.feature[0]][self.feature[1]].squeeze(axis=0).copy()

        height, width, nchannels = bands.shape

        bands = bands.reshape(height * width, nchannels)
        bands_new = np.zeros((height * width, len(HSI_CW)))

        for idx in tqdm(np.arange(height * width)):
            temp_bands = bands[idx]
            temp_mask = np.isfinite(temp_bands)

            if np.sum(temp_mask) > 2:
                spline = CubicSpline(np.array(S2_CW)[temp_mask], temp_bands[temp_mask])
                bands_new[idx, :] = spline(HSI_CW)
            else:
                bands_new[idx, :] = np.zeros(len(HSI_CW), dtype=temp_bands.dtype)

        eopatch[self.feature[0]][self.feature[2]] = bands_new.reshape(
            1, height, width, len(HSI_CW)
        )

        return eopatch


class ResizeTask(EOTask):
    def __init__(
        self,
        features: FeaturesSpecification,
        pixel_size_old: float,
        pixel_size_new: float,
    ):
        self.features = self.parse_renamed_features(features)
        self.pixel_size_old = pixel_size_old
        self.pixel_size_new = pixel_size_new

    def execute(self, eopatch):
        eopatch_new = EOPatch(timestamp=eopatch.timestamp, meta_info=eopatch.meta_info)

        height, width = eopatch.get_spatial_dimension(
            self.features[0][0], self.features[0][1]
        )

        height_new = height * self.pixel_size_old / self.pixel_size_new
        width_new = width * self.pixel_size_old / self.pixel_size_new

        new_size = (int(height_new), int(width_new))

        for feat_type, feat_name, feat_name_new in self.features:
            eopatch_new[feat_type][feat_name_new] = resize_images(
                eopatch[feat_type][feat_name],
                new_size=new_size,
                interpolation="cubic",
            )

        eopatch_new.bbox = eopatch.bbox

        return eopatch_new


class HSICalculationTask(EOTask):
    def __init__(
        self,
        input_feature: Tuple[FeatureType, str],
        output_feature: Tuple[FeatureType, str],
        executable: str,
        calculation: str,
    ):
        """Wapper task to simulate SNR and PSF noise using an external executable binary.

        :param input_feature: Input feature holding radiance values.
        :param output_feature: Output feature with SNR/PSF simulated values.
        :param executable: Path to executable binary.
        :param calculation: Which simulation to execute, i.e. "SNR" or "PSF".
        """
        self.input_feature = input_feature
        self.output_feature = output_feature
        self.executable = executable
        self.calculation = calculation

    def execute(self, eopatch: EOPatch) -> EOPatch:
        with tempfile.TemporaryDirectory(
            prefix="hsi_calc", suffix=self.calculation
        ) as temp_dir:
            input_npy = os.path.join(temp_dir, "input.npy")
            output_npy = os.path.join(temp_dir, "output.npy")

            # temporary write input feature to numpy
            np.save(input_npy, eopatch[self.input_feature])

            # run exec
            subprocess.run(
                f"{self.executable} {self.calculation} {input_npy} {output_npy}".split(
                    " "
                ),
                check=True,
            )

            # read temp output
            eopatch[self.output_feature] = np.load(output_npy)

        # return eopatch with new feature
        return eopatch


def get_shifts(direction: Tuple[float, float]) -> List[Tuple[float, float]]:
    """Compute random shifts."""

    mis_amplitude = np.random.normal(RAND_MEAN, RAND_STD, size=(len(HSI_BANDS),))
    mis_angle = np.random.uniform(low=0, high=2 * np.pi, size=(len(HSI_BANDS),))

    shifts = (mis_amplitude * (np.cos(mis_angle), np.sin(mis_angle))).T + np.array(
        direction
    ) * RELATIVE_SHIFT
    shifts[0, :] = np.array([0.0, 0.0])

    shifts = np.cumsum(shifts, axis=0)

    return [tuple(s) for s in shifts]


class BandMisalignmentTask(EOTask):
    def __init__(
        self,
        feature: FeaturesSpecification,
        interpolation_method: int = INTER_LINEAR,
        border_value: float = 0.0,
    ):
        """Task for simulating band-to-band misalignment

        :param feature: Input feature holding the bands to misalign and the name of the new feature holding the misaligned bands.
        :param interpolation_method: Interpolation method used for band misalignment. Defaults to Linear.
        :param border_value: Constant value use to pad the misaligned bands. Defaults to 0.
        """
        self.input_feature = self.parse_renamed_feature(feature)
        self.interpolation_method = interpolation_method
        self.border_value = border_value

    def execute(self, eopatch: EOPatch) -> EOPatch:
        feat_type, feat_name, new_feat_name = self.input_feature
        eopatch[feat_type][new_feat_name] = np.zeros_like(eopatch[feat_type][feat_name])
        shift_dict = {}

        direction = random.choice(DIRECTIONS)

        for ts_idx, eop_ts in enumerate(eopatch[feat_type][feat_name]):
            warp_matrix = np.eye(3)[:2, :]

            shift_vectors = get_shifts(direction)

            shift_dict[ts_idx] = shift_vectors

            bands_shifted = []

            for b_idx in range(eop_ts.shape[-1]):
                eop_ts_band = eop_ts[..., b_idx]
                warp_matrix[:, 2] = shift_vectors[b_idx]

                band = warpAffine(
                    src=eop_ts_band,
                    M=warp_matrix,
                    dsize=eop_ts_band.T.shape,
                    flags=self.interpolation_method,
                    borderMode=BORDER_CONSTANT,
                    borderValue=self.border_value,
                )

                bands_shifted.append(band[..., np.newaxis])

            eopatch[feat_type][new_feat_name][ts_idx] = np.concatenate(
                bands_shifted, axis=-1
            )

        eopatch.meta_info["Shifts"] = shift_dict
        return eopatch


class GriddingTask(EOTask):
    def __init__(
        self,
        raster_feature: Tuple[FeatureType, str],
        data_stack_feature: Tuple[FeatureType, str],
        grid_feature: Tuple[FeatureType, str],
        size: int,
        overlap: float,
        resolution: float,
        time_index: int = 0,
    ):
        """Split the AOI into smaller image chips to create an AI-ready dataset.

        :param raster_feature: A data feature to crop
        :param grid_feature: A vector feature where cropped grid is saved at.
        :param data_stack_feature: A data feature where output stack of data is stored.
        :param size: A size of of images to crop out of input image.
        :param overlap: Overlap between sub-images extracted.
        :param resolution: Resolution on which task is running.
        """
        self.raster_feature = raster_feature
        self.grid_feature = grid_feature
        self.data_stack_feature = data_stack_feature
        self.size = size
        self.overlap = overlap
        self.resolution = resolution
        self.time_index = time_index

    def _grid_data(self, data: np.array) -> Tuple[List[np.array], Dict]:
        height, width, bands = data.shape
        stride = int(self.size * (1 - self.overlap))

        gridded_data = []
        stats = defaultdict(list)
        for x in range(0, width, stride):
            for y in range(0, height, stride):
                x2, y2 = min(x + self.size, width), min(y + self.size, height)
                x1, y1 = max(0, x2 - self.size), max(0, y2 - self.size)

                if x1 == x2 or y1 == y2:
                    continue

                data_slice = data[y1:y2, x1:x2, ...]
                gridded_data.append(data_slice)

                polygon = shapely.geometry.box(x1, y1, x2, y2)
                stats["pixel_geometry"].append(polygon)

        gridded_data = (
            np.stack(gridded_data, axis=0)
            if gridded_data
            else np.zeros((0, self.size, self.size, bands), dtype=data.dtype)
        )
        return gridded_data, stats

    def execute(self, eopatch: EOPatch) -> EOPatch:
        data = eopatch[self.raster_feature][self.time_index]
        gridded_data, stats = self._grid_data(data)

        eopatch[self.data_stack_feature] = gridded_data

        if self.grid_feature:
            crop_grid_gdf = self.calculate_crop_grid(eopatch, stats)
            eopatch[self.grid_feature] = crop_grid_gdf

        return eopatch

    def calculate_crop_grid(self, eopatch: EOPatch, stats: Dict) -> gpd.GeoDataFrame:
        transform = eopatch.bbox.get_transform_vector(self.resolution, self.resolution)

        def pixel_to_utm_transformer(column, row):
            return pixel_to_utm(row, column, transform=transform)

        utm_polygons = [
            shapely.ops.transform(pixel_to_utm_transformer, polygon)
            for polygon in stats["pixel_geometry"]
        ]
        crop_grid_gdf = gpd.GeoDataFrame(
            stats, geometry=utm_polygons, crs=eopatch.bbox.crs.pyproj_crs()
        )
        return crop_grid_gdf


def get_extent(eopatch: EOPatch) -> Tuple[float, float, float, float]:
    """Calculate the extent (bounds) of the patch.
    :param eopatch: EOPatch for which the extent is calculated.
    :return: The list of EOPatch bounds (min_x, max_x, min_y, max_y)
    """
    return (
        eopatch.bbox.min_x,
        eopatch.bbox.max_x,
        eopatch.bbox.min_y,
        eopatch.bbox.max_y,
    )


class ExportGridToTiff(EOTask):
    def __init__(
        self,
        data_stack_feature: Tuple[FeatureType, str],
        grid_feature: Tuple[FeatureType, str],
        out_folder: str,
        time_index: int = 0,
    ):
        """Export the image chips as GeoTiff files.

        :param grid_feature: A vector feature where cropped grid is saved at.
        :param data_stack_feature: A data feature where output stack of data is stored.
        :param out_folder: Path to folder where tiff will be saved.
        :time_index: Timestamp index in EOPatch from which the grid was created.
        """
        self.grid_feature = grid_feature
        self.data_stack_feature = data_stack_feature
        self.out_folder = out_folder
        self.time_index = time_index

        self.export_task = ExportToTiffTask(
            feature=(FeatureType.DATA, "TEMP_CELL"),
            folder=self.out_folder,
            date_indices=[self.time_index],
        )

    def execute(self, eopatch: EOPatch) -> EOPatch:
        gridded_data = eopatch[self.data_stack_feature]
        gdf = eopatch[self.grid_feature]

        for n_row, row in gdf.iterrows():
            temp_eop = EOPatch(
                data={"TEMP_CELL": gridded_data[[n_row]]},
                bbox=BBox(row.geometry, crs=eopatch.bbox.crs),
                timestamp=[eopatch.timestamp[self.time_index]],
            )

            timestamp_str = eopatch.timestamp[self.time_index].strftime(
                "%Y-%m-%dT%H-%M-%S"
            )
            bbox_str = f"{int(eopatch.bbox.middle[0])}-{int(eopatch.bbox.middle[1])}"
            utm_crs_str = f"{eopatch.bbox.crs.epsg}"
            self.export_task(
                temp_eop,
                filename=f"{bbox_str}_{utm_crs_str}_{self.grid_feature[1]}_{timestamp_str}_{n_row:03d}.tiff",
            )
            del temp_eop

        return eopatch
