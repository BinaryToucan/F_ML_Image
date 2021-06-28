FROM mcr.microsoft.com/dotnet/core/runtime:3.1 AS base
WORKDIR /app

FROM mcr.microsoft.com/dotnet/core/sdk:3.1 AS build
WORKDIR /src
COPY ["F_ML_Image/F_ML_Image.fsproj", "F_ML_Image/"]
RUN dotnet restore "F_ML_Image/F_ML_Image.fsproj"
COPY . .
WORKDIR "/src/F_ML_Image"
RUN dotnet build "F_ML_Image.fsproj" -c Release -o /app/build

FROM build AS publish
RUN dotnet publish "F_ML_Image.fsproj" -c Release -o /app/publish

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "F_ML_Image.dll"]
