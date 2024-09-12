# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [v1.0.0]

### Changed

-   API changed to enable more streamlined user exerience
-   Zarr data structure is now created within the main clustering function so users only interface with an AnnData object

### Added

-   Tests for all major functions

## [v0.2.0]

### Changed

-   Updated ensemble clustering to include scaling the k hyperparameter range based on dataset size, yielding more consistent performance

## [v0.1.0]

### Added

-   Basic tool and plotting functions used for preparing manuscript
