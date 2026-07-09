from pydantic import BaseModel

class HouseInput(BaseModel):

    Id: int

    MSSubClass: int
    MSZoning: str
    LotFrontage: float
    LotArea: float

    Street: str
    Alley: str
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str

    Neighborhood: str

    Condition1: str
    Condition2: str

    BldgType: str
    HouseStyle: str

    OverallQual: int
    OverallCond: int

    YearBuilt: int
    YearRemodAdd: int

    RoofStyle: str
    RoofMatl: str

    Exterior1st: str
    Exterior2nd: str

    MasVnrType: str
    MasVnrArea: float

    ExterQual: str
    ExterCond: str

    Foundation: str

    BsmtQual: str
    BsmtCond: str
    BsmtExposure: str
    BsmtFinType1: str
    BsmtFinSF1: float
    BsmtFinType2: str
    BsmtFinSF2: float
    BsmtUnfSF: float
    TotalBsmtSF: float

    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: str

    FirstFlrSF: float
    SecondFlrSF: float
    LowQualFinSF: float
    GrLivArea: float

    BsmtFullBath: int
    BsmtHalfBath: int

    FullBath: int
    HalfBath: int

    BedroomAbvGr: int
    KitchenAbvGr: int
    KitchenQual: str

    TotRmsAbvGrd: int

    Functional: str

    Fireplaces: int
    FireplaceQu: str

    GarageType: str
    GarageYrBlt: int
    GarageFinish: str
    GarageCars: int
    GarageArea: float
    GarageQual: str
    GarageCond: str

    PavedDrive: str

    WoodDeckSF: float
    OpenPorchSF: float
    EnclosedPorch: float
    ThreeSsnPorch: float
    ScreenPorch: float

    PoolArea: float
    PoolQC: str

    Fence: str

    MiscFeature: str
    MiscVal: float

    MoSold: int
    YrSold: int

    SaleType: str
    SaleCondition: str